
import easyocr
import torch

from .. import data_type as dtp


class TextReader(easyocr.Reader):
    def __init__(self, model_ckpt:str=None) -> None:
        super().__init__(["ch_tra"], detector=False)
        if model_ckpt:
            ckpt = torch.load(model_ckpt)
            self.recognizer.load_state_dict(ckpt)

    # def recognize(self, img: dtp.DetectedInstance):

    

