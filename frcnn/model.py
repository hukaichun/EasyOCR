import typing as tp


import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

from utils import DetectedImage





class FRCNNDetector(torch.nn.Module):
    def __init__(self, classes:tp.List[str], *, model_path:str=None):
        super().__init__()
        self._fasterrcnn = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        _in_features = self._fasterrcnn.roi_heads.box_predictor.cls_score.in_features
        self._fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(_in_features, len(classes))

        if model_path:
            ckpt = torch.load(model_path)
            self._fasterrcnn.load_state_dict(ckpt["model"])

        self._classes = classes
    
    def forward(self, images:torch.Tensor, targets=None):
        return self._fasterrcnn(images, targets)

    def predict(self, images:torch.Tensor):
        # torch._assert(len(images.shape)==4, "")
        result = []
        with torch.no_grad():
            outs = self._fasterrcnn(images)
        return [
            DetectedImage(image, out, classes=self._classes) for image, out in zip(images, outs)
        ]
