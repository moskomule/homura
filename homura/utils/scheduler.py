from abc import ABCMeta


class Scheduler(metaclass=ABCMeta):
    def set_optimizer(self, optimizer):
        pass
