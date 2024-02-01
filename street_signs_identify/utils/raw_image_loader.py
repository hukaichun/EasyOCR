import typing as tp
import json
import logging
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from matplotlib.patches import Ellipse

import shapely

from .. import data_type as dtp

LOGGER = logging.getLogger(__name__)


def _compute_ratio(row):
    polygon:shapely.Polygon = row["polygon"]
    try:
        xx, yy = polygon.exterior.coords.xy
    except:
        print(row)
        assert False
    min_x, min_y = min(xx), min(yy)
    max_x, max_y = max(xx), max(yy)
    return (max_x-min_x)/(max_y-min_y)

def _extract_box(row):
    polygon:shapely.Polygon = row["polygon"]
    xx, yy = polygon.exterior.coords.xy
    min_x, min_y = min(xx), min(yy)
    max_x, max_y = max(xx), max(yy)
    return int(min_x), int(min_y), int(max_x), int(max_y)


def _load_image(img_path) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    return np.array(image)


def _load_annotation(path:Path):
    assert path.is_file()
    with open(path, 'r') as f:
        return annotation2df(json.load(f))


def annotation2df(annotation:tp.Dict):
    result_df = pd.DataFrame(columns=["filename", "size", "polygon", "category", "word"])
    global_index = 0
    for _, v in annotation.items():
        filename = v["filename"]
        size = v["size"]
        _region_list: tp.List[tp.Dict] = v["regions"]
        for region in _region_list:
            _shape_attributes = region["shape_attributes"]
            if _shape_attributes["name"] == "polygon":
                _coord = list(zip(_shape_attributes["all_points_x"], _shape_attributes["all_points_y"]))
                _coord.append(_coord[0])
                polygon = shapely.Polygon(_coord)
            elif _shape_attributes["name"] == "ellipse":
                ellipse = Ellipse((_shape_attributes["cx"], _shape_attributes["cy"]), _shape_attributes["rx"], _shape_attributes["ry"], angle=_shape_attributes["theta"]) 
                vertices = ellipse.get_verts()     # get the vertices from the ellipse object
                polygon = shapely.Polygon(vertices)
            elif _shape_attributes["name"] == "circle":
                ellipse = Ellipse((_shape_attributes["cx"], _shape_attributes["cy"]), _shape_attributes["r"], _shape_attributes["r"]) 
                vertices = ellipse.get_verts()     # get the vertices from the ellipse object
                polygon = shapely.Polygon(vertices)
            elif _shape_attributes["name"] == "rect":
                min_x = _shape_attributes["x"]
                min_y = _shape_attributes["y"]
                max_x = min_x + _shape_attributes["width"]
                max_y = min_y + _shape_attributes["height"]
                polygon = shapely.geometry.box(min_x, min_y, max_x, max_y)
            else:
                print(f"{filename}--{_shape_attributes['name']}")
                raise NotImplementedError("Unknown shape")
            assert isinstance(polygon, shapely.Polygon)
            # polygon = polygon.buffer(0)
            # assert isinstance(polygon, shapely.Polygon)


            _region_attributes = region["region_attributes"]
            category = _region_attributes["category"]
            word = _region_attributes.get("word", None)
            if word and not re.findall(r'\d+|[\u4e00-\u9fff]+|\*', word):
                word = None

            result_df.loc[global_index, "filename"] = filename
            result_df.loc[global_index, "size"] = size
            result_df.loc[global_index, "polygon"] = polygon
            result_df.loc[global_index, "category"] = category
            result_df.loc[global_index, "word"] = word
            global_index+=1

    empty_word_index = result_df["word"].isna()
    result_df.loc[~empty_word_index, "no_star"] = result_df.loc[~empty_word_index].apply(lambda row: "*" not in row["word"], axis=1)
    result_df["ratio"] = result_df.apply(_compute_ratio, axis=1)
    result_df[["x0", "y0", "x1", "y1"]] = result_df.apply(_extract_box, axis=1, result_type="expand")

    return result_df



class RawImageLoader:
    def __init__(self, path:tp.Union[str, Path], annotation_file:str):
        self._data_path = Path(path)
        self._info = _load_annotation(self._data_path/annotation_file)
        self._remove_not_exist_image()
        self._image_names = list(self._info["filename"].unique())

    def _remove_not_exist_image(self):
        possible_files = self._info["filename"].unique()
        not_exist_file = [filename for filename in possible_files if not (self._data_path/filename).is_file()]
        skip_index = self._info["filename"].isin(not_exist_file)
        self._info = self._info.loc[~skip_index].reset_index(drop=True)

    def __getitem__(self, image_name:str):
        assert image_name in self._image_names
        img_path = self._data_path/image_name
        img_np = _load_image(img_path)
        info = self._info[self._info["filename"] == image_name].copy()
        info["score"] = 1
        info["label"] = info["category"]
        return dtp.DetectedImage(img_np, info.reset_index(drop=True))
    
    def __iter__(self):
        for image_name in self._image_names:
            yield self[image_name]
