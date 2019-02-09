import torch

from homura.metrics import commons

input = torch.tensor([[0, 0, 1],
                      [1, 0, 0],
                      [0, 0, 1],
                      [0, 0, 1]], dtype=torch.float)
target = torch.tensor([2, 0, 0, 1], dtype=torch.long)


def test_accuracy():
    assert commons.accuracy(input, target) == 2 / 4
    assert commons.accuracy(input, target, "sum") == 2


def test_true_positive():
    # class 0: 1
    # class 1: 0
    # class 2: 1
    assert all(commons.true_positive(input, target) == torch.tensor([1, 0, 1]).float())


def test_true_negative():
    # class 0: 2
    # class 1: 3
    # class 2: 1
    assert all(commons.true_negative(input, target) == torch.tensor([2, 3, 1]).float())


def test_false_positive():
    # class 0: 0
    # class 1: 0
    # class 2: 2
    assert all(commons.false_positive(input, target) == torch.tensor([0, 0, 2]).float())


def test_false_negative():
    # class 0: 1
    # class 1: 1
    # class 2: 0
    assert all(commons.false_negative(input, target) == torch.tensor([1, 1, 0]).float())


def test_precision():
    # class 1 is nan
    assert all(commons.precision(input, target)[[0, 2]] == torch.tensor([1 / 1, 1 / 3]))


def test_recall():
    assert all(commons.recall(input, target) == torch.tensor([1 / 2, 0 / 1, 1 / 1]))


def test_specificity():
    assert all(commons.specificity(input, target) == torch.tensor([2 / 2, 3 / 3, 1 / 3]))
