import typing as tp
import pandas as pd
import numpy as np
from pandas.core.api import DataFrame as DataFrame

# from ..data_type.detected_image import DetectedImage
from .detector import Detector
from .fr_detector import FRDetector
from .craft_detector import CRAFTDetector
from .. import data_type as dtp



class TowStepDetector(Detector):
    def __init__(self, fr_detector:FRDetector, craft_detector:CRAFTDetector):
        self._fr_detector = fr_detector
        self._craft_detector = craft_detector

    def __call__(self, *images:np.ndarray)-> tp.Union[tp.List[dtp.DetectedImage], dtp.DetectedImage]:
        fr_images = self._fr_detector(*images)
        if not isinstance(fr_images, list):
            fr_images = [fr_images]
        
        result = [self._craft_detection(fr_image) for fr_image in fr_images if fr_image is not None]
        if len(result) == 1:
            return result[0]
        elif len(result) == 0:
            return None
        return result

    def _detection_flow(self, images: np.ndarray) -> tp.List[DataFrame]:
        pass

    def _craft_detection(self, fr_image: dtp.DetectedImage):
        new_df = pd.DataFrame(columns=["label", "score", "x0", "y0", "x1", "y1", "frcnn_label", "frcnn_score"])
        
        for idx in fr_image.info.index:
            fr_instance = fr_image[idx]
            craft_image = self._craft_detector(fr_instance.image)
            if craft_image is None:
                continue
        
            x0_ori, y0_ori, _, _ = fr_instance.bbox
            craft_info = craft_image.info
            craft_info["x0"] += x0_ori
            craft_info["x1"] += x0_ori
            craft_info["y0"] += y0_ori
            craft_info["y1"] += y0_ori
            craft_info["frcnn_label"] = fr_instance.info
            craft_info["frcnn_score"] = fr_instance.score
            new_df = pd.concat([new_df,craft_info], ignore_index=True)
        
        return dtp.DetectedImage(fr_image._ref_image, new_df)
    
