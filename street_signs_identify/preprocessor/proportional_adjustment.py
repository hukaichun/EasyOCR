import typing as tp

import numpy as np

from ..data_type.detected_image import DetectedImage


def proportional_adjustment(ImageHandler:DetectedImage, *index:str, ratio:float):
    adjusted_result = DetectedImage(ImageHandler._ref_image, ImageHandler.info.copy())
    for idx in index:
        h = adjusted_result.info.loc[idx, "y1"] - ImageHandler.info.loc[idx, "y0"]
        h *= ratio
        adjusted_result.info.loc[idx, "y1"] = int(h + adjusted_result.info.loc[idx, "y0"])
        adjusted_result.info.loc[idx, "polygon"] = None
    return adjusted_result

