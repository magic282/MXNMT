from abc import abstractmethod, ABCMeta


class BaseCell(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply(self):
        raise NotImplementedError
