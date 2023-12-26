import dataclasses as dc
import typing as tp
import copy

import torch
from torchvision.ops import nms 
import numpy as np
import pandas as pd
import cv2


def plot_bbox(image:np.ndarray, bbox_list:tp.List[np.ndarray], thickness=1) -> np.ndarray:
    _image = image.copy()
    for bbox in bbox_list:
        x0y0, x1y1 = bbox[:2], bbox[2:]
        # print(x0y0, x1y1, _image.shape)
        cv2.rectangle(_image, x0y0.astype(int), x1y1.astype(int), color=(255,0,0), thickness=thickness)
    return _image

@dc.dataclass
class DetectedInstance:
    bbox: np.ndarray
    category: tp.Union[str, int]
    score: float
    _ref_image: tp.Union[np.ndarray, torch.Tensor] #(H, W, C) (RGB)

    def __post_init__(self):
        if isinstance(self.bbox, torch.Tensor):
            bbox = self.bbox.detach().cpu().numpy()
            self.bbox = np.round(bbox).astype(int)
        # if not isinstance(self._ref_image, np.ndarray):

    @property
    def image(self):
        x0, y0, x1, y1 = self.bbox
        if isinstance(self._ref_image, np.ndarray):
            return self._ref_image[y0:y1, x0:x1].astype(np.uint8)
        else:
            _ref_image_np = self._ref_image.numpy()
            _ref_image_np = _ref_image_np.transpose((1,2,0))
            return _ref_image_np[y0:y1, x0:x1].astype(np.uint8)
        
    @property
    def ref_image(self):
        if isinstance(self._ref_image, np.ndarray):
            return self._ref_image.astype(np.uint8)
        else:
            _ref_image_np = self._ref_image.numpy()
            _ref_image_np = _ref_image_np.transpose((1,2,0))
            return _ref_image_np.astype(np.uint8)


def parse_frcnn_out(frcnn_out:tp.Dict):
    df = pd.DataFrame(columns=["label", "score", "x0", "y0", "x1", "y1"])
    df["label"] = frcnn_out["labels"].numpy()
    df.loc[:, "score"] = frcnn_out["scores"].numpy()
    df[["x0", "y0", "x1", "y1"]] = frcnn_out["boxes"].numpy().round().astype(int)
    return df

class DetectedImage:
    def __init__(self, ref_image, frcnn_out, *, classes:tp.List[str]=None) -> None:
        self._ref_image = ref_image
        self._info = parse_frcnn_out(frcnn_out)
        if classes:
            self._info["label"] = self._info.apply(lambda row: classes[row["label"]], axis=1)

    def __getitem__(self, idx) -> DetectedInstance:
        assert idx in self._info.index, f"{idx} not in {self._info.index}"
        category = self._info.loc[idx, "label"]
        score = self._info.loc[idx, "score"]
        bbox = self._info.loc[idx, ["x0", "y0", "x1", "y1"]]
        return DetectedInstance(bbox, category, score, self._ref_image)

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
    
    def nms_filtering(self, iou_thr=.3):
        keeped_dgs = []
        for label, dg in self._info.groupby("label"):
            bboxes = torch.from_numpy(dg[["x0", "y0", "x1", "y1"]].astype(float).to_numpy())
            scores = torch.from_numpy(dg["score"].astype(float).to_numpy())
            keeped_index = nms(bboxes, scores, iou_thr)
            keeped_dgs.append(dg.iloc[keeped_index])
        keeped_dg = pd.concat(keeped_dgs).reset_index(drop=True)
        new_instance = copy.copy(self)
        new_instance._info = keeped_dg
        return new_instance



    @property
    def ref_image(self):
        if isinstance(self._ref_image, np.ndarray):
            return self._ref_image.astype(np.uint8)
        else:
            _ref_image_np = self._ref_image.numpy()
            _ref_image_np = _ref_image_np.transpose((1,2,0))
            return _ref_image_np.astype(np.uint8)
