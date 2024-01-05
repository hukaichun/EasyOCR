import numpy as np
import pandas as pd
import shapely


def check_overlapping_area(info_this:pd.DataFrame, info_that:pd.DataFrame):
    this_indices, that_indices = check_box_overlap(info_this, info_that)
    if len(this_indices) == 0:
        return None
    overlapping_info = check_poly_overlap(info_this, this_indices, info_that, that_indices)
    overlapping_indices = overlapping_info["area"]>0
    return overlapping_info.loc[overlapping_indices].copy()


def check_box_overlap(info_this:pd.DataFrame, info_that:pd.DataFrame):
    x0y0_this = info_this.loc[:, ["x0", "y0"]].to_numpy()
    x1y1_this = info_this.loc[:, ["x1", "y1"]].to_numpy()
    x0y0_that = info_that.loc[:, ["x0", "y0"]].to_numpy()
    x1y1_that = info_that.loc[:, ["x1", "y1"]].to_numpy()
    
    x0y0_over = np.maximum(x0y0_this[:, np.newaxis], x0y0_that[np.newaxis])
    x1y1_over = np.minimum(x1y1_this[:, np.newaxis], x1y1_that[np.newaxis])
    legal_box = np.all(x1y1_over > x0y0_over, axis=2)
    idx_this, idx_that = np.where(legal_box)

    index_this = info_this.index[idx_this]
    index_that = info_that.index[idx_that]
    
    return index_this, index_that


def check_poly_overlap(info_this:pd.DataFrame, index_this:pd.Index,
                       info_that:pd.DataFrame, index_that:pd.Index):
    info_this, info_that = info_this.copy(), info_that.copy()
    if "polygon" not in info_this:
        info_this["polygon"] = None
    if "polygon" not in info_that:
        info_that["polygon"] = None

    index_this_unique = index_this.unique()
    _no_polygon_this_ith = info_this.loc[index_this_unique, "polygon"].isna()
    _no_polygon_this_idx = index_this_unique[_no_polygon_this_ith]
    if len(_no_polygon_this_idx) > 0:
        new_polygon = info_this.loc[_no_polygon_this_idx].apply(lambda row: shapely.geometry.box(row["x0"], row["y0"], row["x1"], row["y1"]), axis=1)
        info_this.loc[_no_polygon_this_idx, "polygon"] = new_polygon
    
    index_that_unique = index_that.unique()
    _no_polygon_that_ith = info_that.loc[index_that_unique, "polygon"].isna()
    _no_polygon_that_idx = index_that_unique[_no_polygon_that_ith]
    if len(_no_polygon_that_idx) > 0:
        new_polygon = info_that.loc[_no_polygon_that_idx].apply(lambda row: shapely.geometry.box(row["x0"], row["y0"], row["x1"], row["y1"]), axis=1)
        info_that.loc[_no_polygon_that_idx, "polygon"] = new_polygon

    result_df = pd.DataFrame(columns=["index_this", "index_that", "area"])
    result_df["index_this"] = index_this
    result_df["index_that"] = index_that

    def comput_overlap_area(poly_this:shapely.Polygon, poly_that:shapely.Polygon):
        poly_overlap = poly_this.intersection(poly_that)
        if poly_overlap.is_empty:
            return 0.
        else:
            return poly_overlap.area

    result_df["area"] = result_df.apply(lambda row: comput_overlap_area(info_this.loc[row["index_this"], "polygon"], info_that.loc[row["index_that"], "polygon"] ), axis=1)

    return result_df