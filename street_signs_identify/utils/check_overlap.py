import numpy as np
import pandas as pd
import shapely


def check_overlapping_area(info_this:pd.DataFrame, info_that:pd.DataFrame):
    this_indices, that_indices = check_box_overlap(info_this, info_that)
    if len(this_indices) == 0:
        return None
    overlapping_info = check_poly_overlap(info_this, this_indices, info_that, that_indices)
    overlapping_info = overlapping_info.loc[overlapping_info["intersection_area"]>0].copy()
    overlapping_info["IOU"] = overlapping_info["intersection_area"]/overlapping_info["union_area"]
    return overlapping_info


def check_box_overlap(info_1st:pd.DataFrame, info_2nd:pd.DataFrame):
    x0y0_this = info_1st.loc[:, ["x0", "y0"]].to_numpy()
    x1y1_this = info_1st.loc[:, ["x1", "y1"]].to_numpy()
    x0y0_that = info_2nd.loc[:, ["x0", "y0"]].to_numpy()
    x1y1_that = info_2nd.loc[:, ["x1", "y1"]].to_numpy()
    
    x0y0_over = np.maximum(x0y0_this[:, np.newaxis], x0y0_that[np.newaxis])
    x1y1_over = np.minimum(x1y1_this[:, np.newaxis], x1y1_that[np.newaxis])
    legal_box = np.all(x1y1_over > x0y0_over, axis=2)
    idx_this, idx_that = np.where(legal_box)

    index_this = info_1st.index[idx_this]
    index_that = info_2nd.index[idx_that]
    
    return index_this, index_that


def check_poly_overlap(info_1st:pd.DataFrame, index_1st:pd.Index,
                       info_2nd:pd.DataFrame, index_2nd:pd.Index):
    info_1st, info_2nd = info_1st.copy(), info_2nd.copy()
    if "polygon" not in info_1st:
        info_1st["polygon"] = None
    if "polygon" not in info_2nd:
        info_2nd["polygon"] = None

    index_this_unique = index_1st.unique()
    _no_polygon_this_ith = info_1st.loc[index_this_unique, "polygon"].isna()
    _no_polygon_this_idx = index_this_unique[_no_polygon_this_ith]
    if len(_no_polygon_this_idx) > 0:
        new_polygon = info_1st.loc[_no_polygon_this_idx].apply(lambda row: shapely.geometry.box(row["x0"], row["y0"], row["x1"], row["y1"]), axis=1)
        info_1st.loc[_no_polygon_this_idx, "polygon"] = new_polygon
    
    index_that_unique = index_2nd.unique()
    _no_polygon_that_ith = info_2nd.loc[index_that_unique, "polygon"].isna()
    _no_polygon_that_idx = index_that_unique[_no_polygon_that_ith]
    if len(_no_polygon_that_idx) > 0:
        new_polygon = info_2nd.loc[_no_polygon_that_idx].apply(lambda row: shapely.geometry.box(row["x0"], row["y0"], row["x1"], row["y1"]), axis=1)
        info_2nd.loc[_no_polygon_that_idx, "polygon"] = new_polygon

    result_df = pd.DataFrame(columns=["index_1st", "index_2nd", "intersection_area", "union_area"])
    result_df["index_1st"] = index_1st
    result_df["index_2nd"] = index_2nd

    def comput_intersection_area(poly_this:shapely.Polygon, poly_that:shapely.Polygon):
        ## 檢查多邊形是否有效
        if not poly_this.is_valid or not poly_that.is_valid:
            ## 修復無效多邊形
            poly_this = poly_this.buffer(0)
            poly_that = poly_that.buffer(0)
    
        poly_overlap = poly_this.intersection(poly_that)
        if poly_overlap.is_empty:
            return 0.
        else:
            return poly_overlap.area
        
    def comput_union_area(poly_this:shapely.Polygon, poly_that:shapely.Polygon):
        ## 檢查多邊形是否有效
        if not poly_this.is_valid or not poly_that.is_valid:
            #@ 修復無效多邊形
            poly_this = poly_this.buffer(0)
            poly_that = poly_that.buffer(0)
            
        _poly_union = shapely.union(poly_this, poly_that)
        poly_union = shapely.normalize(_poly_union)
        
        return poly_union.area

    result_df["intersection_area"] = result_df.apply(lambda row: comput_intersection_area(info_1st.loc[row["index_1st"], "polygon"], info_2nd.loc[row["index_2nd"], "polygon"] ), axis=1)
    result_df["union_area"] = result_df.apply(lambda row: comput_union_area(info_1st.loc[row["index_1st"], "polygon"], info_2nd.loc[row["index_2nd"], "polygon"] ), axis=1)

    return result_df