import abc


class FTS(abc.ABC):

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def predict(self):
        pass
