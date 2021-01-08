import abc


class BaseGenerator(object, metaclass=abc.ABCMeta):

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    @abc.abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def batchify(self):
        return None
