import abc

import easyocr
import torch

from .. import data_type as dtp


class Recognizer(abc.ABC):
    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        # self._reader = easyocr.Reader(["ch_tra"], detector=False)
        # if model_ckpt:
        #     ckpt = torch.load(model_ckpt)
        #     self._reader.recognizer.load_state_dict(ckpt)

    @abc.abstractmethod
    def _recognize(self, detected_instance:dtp.DetectedInstance) -> dtp.DetectedInstance:
        pass

    def __call__(self, *detected_instances: dtp.DetectedInstance):

        # check all inputs are all valid
        for detected_instance in detected_instances:
            assert isinstance(detected_instance, dtp.DetectedInstance)

        return [
            self._recognize(instance) for instance in detected_instances
        ]