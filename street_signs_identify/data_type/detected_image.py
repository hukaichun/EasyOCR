import typing as tp

import torch
from torchvision.ops import nms 
import numpy as np
import pandas as pd
import cv2

from .detected_instance import DetectedInstance



def plot_bbox(image:np.ndarray, bbox_list:tp.List[np.ndarray], thickness=1) -> np.ndarray:
    _image = image.copy()
    for bbox in bbox_list:
        x0y0, x1y1 = bbox[:2], bbox[2:]
        # print(x0y0, x1y1, _image.shape)
        cv2.rectangle(_image, x0y0.astype(int), x1y1.astype(int), color=(255,0,0), thickness=thickness)
    return _image


class DetectedImage:
    def __init__(self, ref_image: tp.Union[np.ndarray, torch.Tensor], info: pd.DataFrame) -> None:
        self._ref_image = ref_image
        self._info = info

    def __getitem__(self, idx) -> DetectedInstance:
        assert idx in self._info.index, f"{idx} not in {self._info.index}"
        label = self._info.loc[idx, "label"]
        score = self._info.loc[idx, "score"]
        bbox = self._info.loc[idx, ["x0", "y0", "x1", "y1"]]
        return DetectedInstance(bbox, label, score, self._ref_image)

    def fetch(self, label:tp.Union[str, int], *, score_thr=.6)-> tp.List[DetectedInstance]:
        check_label = self._info["label"] == label
        check_score = self._info["score"] > score_thr
        check = check_label & check_score
        indices = self._info.index[check]
        return [self[idx] for idx in indices]

    def draw_box(self, label:tp.Union[str, int], *, thickness=3, score_thr=.6):
        check_label = self._info["label"] == label
        check_score = self._info["score"] > score_thr
        check = check_label & check_score
        bboxes = self._info.loc[check, ["x0", "y0", "x1", "y1"]].to_numpy()
        image = self.ref_image
        return plot_bbox(image, bboxes, thickness=thickness)
    
    def draw_all_box(self, thickness=3, score_thr=.6):
        check_score = self._info["score"] > score_thr
        bboxes = self._info.loc[check_score, ["x0", "y0", "x1", "y1"]].to_numpy()
        image = self.ref_image
        return plot_bbox(image, bboxes, thickness=thickness)
    
    def nms_filtering(self, iou_thr=.3) -> "DetectedImage":
        keeped_dgs = []
        for _, dg in self._info.groupby("label"):
            bboxes = torch.from_numpy(dg[["x0", "y0", "x1", "y1"]].astype(float).to_numpy())
            scores = torch.from_numpy(dg["score"].astype(float).to_numpy())
            keeped_index = nms(bboxes, scores, iou_thr)
            keeped_dgs.append(dg.iloc[keeped_index])
        keeped_dg = pd.concat(keeped_dgs).reset_index(drop=True)
        new_instance = DetectedImage(self._ref_image, keeped_dg)
        return new_instance

    @property
    def ref_image(self):
        if isinstance(self._ref_image, np.ndarray):
            return self._ref_image.astype(np.uint8)
        else:
            _ref_image_np = self._ref_image.numpy()
            _ref_image_np = _ref_image_np.transpose((1,2,0))
            return _ref_image_np.astype(np.uint8)


class FakeDetectedImage(DetectedImage):
    def __init__(self, ref_image: tp.Union[np.ndarray, torch.Tensor],) -> None:
        super().__init__(ref_image, pd.DataFrame(columns=["label", "score", "x0", "y0", "x1", "y1"]))
        h, w, c = self.ref_image.shape
        self._info.loc[0, "label"] = "None"
        self._info.loc[0, "score"] = 1
        self._info.loc[0, "x0"] = 0
        self._info.loc[0, "y0"] = 0
        self._info.loc[0, "x1"] = w
        self._info.loc[0, "y1"] = h