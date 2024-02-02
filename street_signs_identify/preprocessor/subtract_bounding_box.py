import typing as tp

import cv2
import shapely
import numpy as np

from ..data_type.detected_image import DetectedImage
from ..utils import check_overlap as co


def interval_subtraction(interval1:tp.Tuple[int, int], interval2:tp.Tuple[int, int]):
    interval1 = np.atleast_1d(interval1)
    interval2 = np.atleast_1d(interval2)
    intersect = np.array([max(interval1[0], interval2[0]), min(interval1[1], interval2[1])])
    
    if np.all(intersect == interval1):
        return None

    if intersect[1] - intersect[0] < 0:
        # not overlapping; do nothing
        return interval1
    
    if interval1[0]<intersect[0] and interval1[1]>intersect[1]:
        # interval2 inside the interval1
        piece1 = np.array([interval1[0], intersect[0]])
        piece2 = np.array([intersect[1], interval1[1]])
        if piece1[1]-piece1[0] > piece2[1]-piece2[0]:
            return piece1
        else:
            return piece2
        
    if interval1[0]<intersect[0]:
        return np.array([interval1[0], intersect[0]])
    else:
        return np.array([intersect[1], interval1[1]])
    

def subtract_bounding_box(ImageHandler:DetectedImage, *category:str, ref_ImageHandler:DetectedImage=None):
    all_categories = category

    if ref_ImageHandler:
        in_categories_index = ref_ImageHandler.info["category"].isin(all_categories)
        in_categories_df = ref_ImageHandler.info.loc[in_categories_index]
        out_of_categories_df = ImageHandler.info
    else:
        in_categories_index = ImageHandler.info["category"].isin(all_categories)
        out_of_categories_index = ~in_categories_index
        in_categories_df = ImageHandler.info.loc[in_categories_index]
        out_of_categories_df = ImageHandler.info.loc[out_of_categories_index]

    result = co.check_overlapping_area(out_of_categories_df, in_categories_df)
    if result is None: return ImageHandler
    
    vanish_index_out = []
    for index_out, dg in result.groupby("index_1st"):
        out_of_categories_df.loc[index_out, "polygon"] = None
        for index_in in dg["index_2nd"]:
            interval1 = out_of_categories_df.loc[index_out, "x0"], out_of_categories_df.loc[index_out, "x1"]
            interval2 = in_categories_df.loc[index_in, "x0"], in_categories_df.loc[index_in, "x1"]
            interval3 = interval_subtraction(interval1, interval2)
            if interval3 is None:
                vanish_index_out.append(index_out)
            else:
                # print("subtraction success")
                x0, x1 = interval3
                assert x1>x0
                out_of_categories_df.loc[index_out, "x0"] = x0
                out_of_categories_df.loc[index_out, "x1"] = x1

    if vanish_index_out:
        out_of_categories_df = out_of_categories_df.drop(index=vanish_index_out)
    return DetectedImage(ImageHandler._ref_image, out_of_categories_df)
    



