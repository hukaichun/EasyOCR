import abc

from .. import data_type as dtp


class Recognizer(abc.ABC):
    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)

    @abc.abstractmethod
    def _recognize(self, detected_instance:dtp.DetectedInstance) -> dtp.DetectedInstance:
        pass

    def __call__(self, *detected_instances: dtp.DetectedInstance):

        # check all inputs are all valid
        for detected_instance in detected_instances:
            assert isinstance(detected_instance, dtp.DetectedInstance)

        result = [
            self._recognize(instance) for instance in detected_instances
        ]

        if len(result) == 1:
            return result[0]

        return result