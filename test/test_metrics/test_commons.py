import torch

from homura.metrics import commons

# pred: [2, 0, 2, 2]
input = torch.tensor([[0, 0, 1],
                      [1, 0, 0],
                      [0, 0, 1],
                      [0, 0, 1]], dtype=torch.float)
target = torch.tensor([2, 0, 0, 1], dtype=torch.long)


def test_confusion_matrix():
    cm = commons.confusion_matrix(input, target)
    expected = torch.zeros(3, 3, dtype=torch.long)
    expected[2, 2] = 1
    expected[0, 0] = 1
    expected[2, 0] = 1
    expected[2, 1] = 1
    assert all(cm.view(-1) == expected.view(-1))


def test_classwise_accuracy():
    assert all(commons.classwise_accuracy(input, target) == torch.tensor([3 / 4, 3 / 4, 2 / 4]))


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


def test_true_positive_2d():
    # torch.randn(2, 3, 2, 2)
    input = torch.tensor([[[[0.0146, 0.8026],
                            [0.5576, -2.3168]],

                           [[-1.1490, 0.6365],
                            [-1.1506, -0.6319]],

                           [[-0.4976, 0.8760],
                            [0.6989, -1.1562]]],

                          [[[-0.0541, -0.0892],
                            [-0.9677, 1.3331]],

                           [[1.7848, 1.0078],
                            [0.7506, -1.5101]],

                           [[-0.6134, 1.9541],
                            [1.1825, -0.5879]]]])

    # argmax(dim=1)
    target = torch.tensor([[[0, 2],
                            [2, 1]],

                           [[1, 2],
                            [2, 0]]])

    assert all(commons.true_positive(input, target) == torch.tensor([2, 2, 4]).float())


def test_accuracy():
    input = torch.tensor([[0.9159, -0.3400, -1.0952, 0.1969, 0.4769],
                          [-0.1677, 0.7205, 0.3802, -0.8408, 0.5447],
                          [0.1596, 0.0366, -1.3719, 1.6869, -0.2422]])
    # argmax=[0, 1, 3], argmin=[2, 3, 2]
    target = torch.tensor([0, 3, 0])
    assert commons.accuracy(input, target) == torch.tensor([1 / 3])
    assert commons.accuracy(input, target, top_k=3) == torch.tensor([2 / 3])
    assert commons.accuracy(input, target, top_k=5) == torch.tensor([1.0])
