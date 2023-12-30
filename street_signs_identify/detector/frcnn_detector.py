import typing as tp
from typing import Dict
from numpy import ndarray

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .detector import DetectorBase

class FRCNNDetector(torch.nn.Module):
    def __init__(self, classes:tp.List[str], *, model_path:str=None):
        super().__init__()
        self._fasterrcnn = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        _in_features = self._fasterrcnn.roi_heads.box_predictor.cls_score.in_features
        self._fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(_in_features, len(classes))

        if model_path:
            ckpt = torch.load(model_path)
            self.load_state_dict(ckpt)

        self._classes = classes
    
    def forward(self, images:torch.Tensor, targets=None):
        return self._fasterrcnn(images, targets)


class FrcnnDetector(DetectorBase):
    def __init__(self, classes:tp.List[str], *, model_path:str=None) -> None:
        super().__init__()
        self._detector = FRCNNDetector(classes, model_path)
        self._detector.eval()

    def _detection_flow(self, images: ndarray) -> Dict:
        with torch.no_grad:
            return self._detector(images)