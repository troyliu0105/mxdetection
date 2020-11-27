import abc


class AbstractTransformer(object, metaclass=abc.ABCMeta):
    def __call__(self, *args):
        assert len(args) == 2
        return self.do(*args)

    @abc.abstractmethod
    def do(self, img, target):
        raise NotImplementedError()
