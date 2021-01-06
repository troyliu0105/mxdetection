import abc


class BaseAssigner(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def extract(self, F, pred_args, label_args):
        pass
