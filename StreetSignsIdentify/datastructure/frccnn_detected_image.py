import typing as tp

import pandas as pd

from .detected_image import DetectedImage


def parse_frcnn_out(frcnn_out:tp.Dict):
    df = pd.DataFrame(columns=["label", "score", "x0", "y0", "x1", "y1"])
    df["label"] = frcnn_out["labels"].numpy()
    df.loc[:, "score"] = frcnn_out["scores"].numpy()
    df[["x0", "y0", "x1", "y1"]] = frcnn_out["boxes"].numpy().round().astype(int)
    return df


class FRCNNDetectedImage(DetectedImage):
    def __init__(self, ref_image, frcnn_out:tp.Dict, *, classes:tp.List[str]=None) -> None:
        _info = parse_frcnn_out(frcnn_out)
        if classes:
            _info["label"] = self._info.apply(lambda row: classes[row["label"]], axis=1)
        super().__init__(ref_image, info=_info)