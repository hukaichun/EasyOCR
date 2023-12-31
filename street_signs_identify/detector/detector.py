import typing as tp
import abc

import pandas as pd
import numpy as np


from .. import data_type as dtp


def parse_dict_out(dict_out:tp.Dict):
    df = pd.DataFrame(columns=["label", "score", "x0", "y0", "x1", "y1"])
    df["label"] = dict_out["labels"].numpy()
    df.loc[:, "score"] = dict_out["scores"].numpy()
    df.loc[:, ["x0", "y0", "x1", "y1"]] = dict_out["boxes"].numpy().round().astype(int)
    return df


class Detector(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *images:np.ndarray)-> tp.Union[tp.List[dtp.DetectedImage], dtp.DetectedImage]:
        # check all inputs are all valid
        for image in images:
            assert isinstance(image, np.ndarray)

        outs: tp.List[tp.Dict] = self._detection_flow(images)
        result = [dtp.DetectedImage(img, parse_dict_out(out)) for img, out in zip(images, outs)]
        if len(result) == 1:
            result=result[0]
        return result

    @abc.abstractmethod
    def _detection_flow(self, images:np.ndarray) -> tp.Dict:
        pass