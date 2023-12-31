import typing as tp
import easyocr
import numpy as np
import pandas as pd

from .detector import Detector


class CRAFTDetector(Detector):
    def __init__(self) -> None:
        super().__init__()

        self._reader = easyocr.Reader(["ch_tra"])
        del self._reader.recognizer

    def _detection_flow(self, images: tp.List[np.ndarray]) -> tp.List[pd.DataFrame]:
        all_dfs = []
        for image in images:
            h_list, f_list = self._reader.detect(image)
            outs_df = []
            for idx, bboxes in enumerate(h_list):
                if bboxes:
                    df = pd.DataFrame(columns=["label", "score", "x0", "y0", "x1", "y1"])
                    bboxes_np = np.asarray(bboxes).squeeze()
                    assert len(bboxes_np.shape)==2
                    df[["x0", "x1", "y0", "y1"]] = bboxes_np
                    df.loc[:, "label"] = "h_list"
                    df.loc[:, "score"] = 1.
                    outs_df.append(df)


            for idx, bboxes in enumerate(f_list):
                if bboxes:
                    df = pd.DataFrame(columns=["label", "score", "x0", "y0", "x1", "y1"])
                    bboxes_np = np.asarray(bboxes).squeeze()
                    assert len(bboxes_np.shape)==2, bboxes_np
                    df[["x0", "x1", "y0", "y1"]] = bboxes_np
                    df.loc[:, "label"] = "f_list"
                    df.loc[:, "score"] = 1.
                    outs_df.append(df)
            full_df = pd.concat(outs_df, axis=0)
            all_dfs.append(full_df)
        return all_dfs