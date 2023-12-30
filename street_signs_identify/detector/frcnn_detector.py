import typing as tp
from typing import Dict
from numpy import ndarray

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .detector import Detector


class FRCNNDetector(torch.nn.Module):
    def __init__(self, classes:tp.List[str], *, model_ckpt:str=None):
        super().__init__()
        self._fasterrcnn = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        _in_features = self._fasterrcnn.roi_heads.box_predictor.cls_score.in_features
        self._fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(_in_features, len(classes))

        if model_ckpt:
            ckpt = torch.load(model_ckpt)
            self.load_state_dict(ckpt)

        self._classes = classes
    
    def forward(self, images:torch.Tensor, targets=None):
        return self._fasterrcnn(images, targets)


class FRCDetector(Detector):
    def __init__(self, *, 
                 classes:tp.List[str]=['1', '2', '3', '4', '5', 'blue', 'brown', 'green', 'red', 'yellow', 'parking', 'limit_h', 'limit_speed' ,'14', '15'], 
                 model_ckpt:str=None,
                 DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        super().__init__()
        self._detector = FRCNNDetector(classes, model_ckpt=model_ckpt)
        self._detector.to(DEVICE)
        self._detector.eval()

        self.__DEVICE = DEVICE


    def _detection_flow(self, images: ndarray) -> Dict:

        images_np = [image.transpose((2,0,1)) for image in images] # H,W,C to C,H,W format
        images_torch = torch.stack([torch.from_numpy(image).float() for image in images_np]).to(self.__DEVICE)

        with torch.no_grad():
            outs_torch = self._detector(images_torch)
        outs = [{k:v.to("cpu") for k,v in out_torch.items()} for out_torch in outs_torch]
        return outs