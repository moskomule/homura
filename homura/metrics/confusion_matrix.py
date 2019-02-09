import torch


class ConfusionMatrix(object):
    def __init__(self, num_classes: int):
        """
        calculate confusion matrix
        :param num_classes:
        """
        self._matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, predictions: torch.Tensor, ground_truths: torch.Tensor):
        for pred, gt in zip(predictions.flatten(), ground_truths.flatten()):
            self._matrix[pred, gt] += 1

    @property
    def numpy(self):
        return self._matrix.numpy()

    @property
    def matrix(self):
        return self._matrix
