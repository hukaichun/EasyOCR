import typing as tp
import abc

import pandas as pd
import numpy as np


from .. import data_type as dtp


def parse_dict_out(dict_out:tp.Dict):
    df = pd.DataFrame(columns=["label", "score", "x0", "y0", "x1", "y1"])
    df["label"] = dict_out["labels"].numpy()
    df.loc[:, "score"] = dict_out["scores"].numpy()
    df[["x0", "y0", "x1", "y1"]] = dict_out["boxes"].numpy().round().astype(int)
    return df


class DetectorBase(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    def detect(self, *images:np.ndarray)-> tp.List[dtp.DetectedImage]:
        outs = self._detection_flow(images)
        return [
            dtp.DetectedImage(img, parse_dict_out(out)) for img, out in zip(images, outs)
        ]

    @abc.abstractmethod
    def _detection_flow(self, images:np.ndarray) -> tp.Dict:
        pass