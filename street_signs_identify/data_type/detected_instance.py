import dataclasses as dc
import typing as tp

import torch
import numpy as np


@dc.dataclass
class DetectedInstance:
    bbox: np.ndarray
    info: tp.Union[str, int]
    score: float
    _ref_image: tp.Union[np.ndarray, torch.Tensor] #(H, W, C) (RGB)

    def __post_init__(self):
        if isinstance(self.bbox, torch.Tensor):
            bbox = self.bbox.detach().cpu().numpy()
            self.bbox = bbox
        self.bbox = np.maximum(self.bbox, 0).astype(float)
        self.bbox = np.round(self.bbox).astype(int)

    @property
    def image(self):
        x0, y0, x1, y1 = self.bbox
        return self.ref_image[y0:y1, x0:x1]
        
    @property
    def ref_image(self) -> np.ndarray:
        if isinstance(self._ref_image, np.ndarray):
            return self._ref_image.astype(np.uint8)
        else:
            _ref_image_np = self._ref_image.numpy()
            _ref_image_np = _ref_image_np.transpose((1,2,0))
            return _ref_image_np.astype(np.uint8)
        
    @property
    def legal(self):
        x0, y0, x1, y1 = self.bbox
        return x1>x0 and y1>y0
