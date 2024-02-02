import typing as tp
import os
import logging

import pandas as pd
from PIL import Image, ImageOps
import numpy as np

import shapely

from .. import data_type as dtp

LOGGER = logging.getLogger(__name__)


def _load_image(img_path) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.exif_transpose(image) # rotation here
    return np.array(image)

def _load_info(labeling_file_path):
    df = pd.read_csv(labeling_file_path, encoding='big5')

    def path_str_to_closedpolygon(path_str):
        list_of_pts: tp.List = eval(path_str)
        if list_of_pts[0] != list_of_pts[-1]:
            list_of_pts.append(list_of_pts[0])
        return shapely.Polygon(list_of_pts)
    df.loc[:, "polygon"] = df.apply(lambda row: path_str_to_closedpolygon(row["box"]), axis=1)

    def polygon_to_boundingbox(poly: shapely.Polygon):
        xs, ys = poly.exterior.xy
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return int(x0), int(y0), int(x1), int(y1)
    
    df[["x0", "y0", "x1", "y1"]] = df.apply(lambda row: polygon_to_boundingbox(row["polygon"]), axis=1, result_type="expand")
    df.rename(columns={"words":"label"}, inplace=True)
    # dg = df.apply(lambda row: row["words"].split("\n"), axis=1, result_type="expand")
    # # df["label_"] = df.apply(lambda row: row["words"].split("\n")[1], axis=1)
    # df = pd.concat([df, dg], axis=1)

    return df



class XImageLoader:
    def __init__(self, data_path:str, annotation_file:str="x_labels.csv"):
        self._data_path = data_path
        self._info = _load_info(os.path.join(data_path, annotation_file))
        self._image_names = list(self._info["raw_filename"].unique())
        self._drop_not_exist_filename()

    def __getitem__(self, image_name:str):
        assert image_name in self._image_names
        img_path = os.path.join(self._data_path, image_name)
        assert os.path.isfile(img_path)
        img_np = _load_image(img_path)
        info = self._info[self._info["raw_filename"] == image_name].copy()
        info["score"] = 1
        return dtp.DetectedImage(img_np, info.reset_index(drop=True))

    def __iter__(self):
        for image_name in self._image_names:
            yield self[image_name]

    def _drop_not_exist_filename(self):
        not_exist_list = []
        for image_name in self._image_names:
            img_path = os.path.join(self._data_path, image_name)
            if not os.path.isfile(img_path):
                LOGGER.warning("Ignore data whose file is not exist: \n%s", img_path)
                not_exist_list.append(image_name)

        for tmp in not_exist_list:
            self._image_names.remove(tmp)