import typing as tp
import abc

import pandas as pd
import numpy as np


from .. import data_type as dtp





class Detector(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *images:np.ndarray)-> tp.Union[tp.List[dtp.DetectedImage], dtp.DetectedImage]:
        # check all inputs are all valid
        for image in images:
            assert isinstance(image, np.ndarray)

        outs: tp.List[pd.DataFrame] = self._detection_flow(images)
        result = [dtp.DetectedImage(img, out) for img, out in zip(images, outs)]
        if len(result) == 1:
            result=result[0]
        return result

    @abc.abstractmethod
    def _detection_flow(self, images:np.ndarray) -> tp.List[pd.DataFrame]:
        pass