import typing as tp

import cv2
import shapely
import numpy as np
import pandas as pd

from ..data_type.detected_image import DetectedImage


def masking_polygon(ImageHandler:DetectedImage, *mask_index:tp.Any, color=(0,0,0)):
    image, new_info = masking_polygon_low_level(ImageHandler.ref_image, ImageHandler.info, *mask_index, color=color)
    result = DetectedImage(image, new_info)
    return result


def masking_polygon_low_level(image:np.ndarray, info:pd.DataFrame, *mask_index:tp.Any, color=(0,0,0)):
    assert "polygon" in info.columns, "Not come from GT?"

    for index in mask_index:
        polygon:shapely.Polygon = info.loc[index, "polygon"]
        all_points_x, all_points_y = polygon.exterior.coords.xy
        polygon_np = np.array(list(zip(all_points_x, all_points_y))).astype(int)
        cv2.drawContours(image, [polygon_np], 0, color, thickness=cv2.FILLED)

    image = image.copy()
    unused_index = info.index.isin(mask_index)
    new_info = info.loc[~unused_index]
    return image, new_info