import easyocr
import torch
import numpy as np

from .recognizer import Recognizer
from .. import data_type as dtp



class EasyOCRRecognizer(Recognizer):
    def __init__(self, model_ckpt:str=None, *, lang_list=["ch_tra"]):
        self._reader = easyocr.Reader(lang_list, detector=False)
        if model_ckpt:
            ckpt = torch.load(model_ckpt)
            self._reader.recognizer.load_state_dict(ckpt)

    def _recognize(self, detected_instance:dtp.DetectedInstance):
        out = self._reader.recognize(detected_instance.image)[0]
        bbox_path = np.asarray(out[0])
        x0,y0 = np.min(bbox_path,axis=0)
        x1,y1 = np.max(bbox_path,axis=0)
        bbox = np.asarray([x0,y0,x1,y1])
        return dtp.DetectedInstance(bbox, out[1], out[2], detected_instance.image)