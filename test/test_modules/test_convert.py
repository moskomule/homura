import torch

from homura.modules.functional import convert


def test_onehot():
    num_classes = 4
    label = torch.tensor([1, 2, 0])
    expected = torch.zeros(3, num_classes)
    expected[0, 1] = 1
    expected[1, 2] = 1
    expected[2, 0] = 1
    assert all(convert.to_onehot(label, num_classes).view(-1) == expected.view(-1))

    label = torch.zeros(1, 2, 2, dtype=torch.long)
    label[0, 0, 0] = 0
    label[0, 0, 1] = 2
    label[0, 1, 0] = 3
    label[0, 1, 1] = 1

    expected = torch.zeros(1, num_classes, 2, 2)
    expected[0, 0, 0, 0] = 1
    expected[0, 2, 0, 1] = 1
    expected[0, 3, 1, 0] = 1
    expected[0, 1, 1, 1] = 1
    assert all(convert.to_onehot(label, num_classes).view(-1) == expected.view(-1))
