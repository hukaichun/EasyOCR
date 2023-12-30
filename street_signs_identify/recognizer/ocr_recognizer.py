
import easyocr
import torch

from .. import data_type as dtp


class TextReader(easyocr.Reader):
    def __init__(self, model_ckpt:str=None) -> None:
        super().__init__(["ch_tra"], detector=False)
        if model_ckpt:
            ckpt = torch.load(model_ckpt)
            self.recognizer.load_state_dict(ckpt)

    def _recognizeDetectedInstance(self, detected_instance:dtp.DetectedInstance):
        out = self.recognize(detected_instance.image)[0]
        return dtp.DetectedInstance(out[0], out[1], out[2], detected_instance._ref_image)

    def recognizeDetectedInstance(self, *detected_instances: dtp.DetectedInstance):
        return [
            self._recognizeDetectedInstance(instance) for instance in detected_instances
        ]
            
    

