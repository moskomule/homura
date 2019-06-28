import pytest
import torch

from homura.metrics import segmentation

binary_input = torch.zeros(1, 1, 3, 3)
binary_input[0, :, :, 0:2] = 1
# tensor([[[1., 1., 0.],
#          [1., 1., 0.],
#          [1., 1., 0.]]])
binary_target = torch.zeros(1, 3, 3, dtype=torch.long)
binary_target[0, :, 1:3] = 1
# tensor([[[0, 1, 1],
#          [0, 1, 1],
#          [0, 1, 1]]])
# So accuracy is 3 / 9 and IoU is 3 / 9

multi_input = torch.zeros(2, 3, 3, 3)
multi_input[0, 2, :, 0:2] = 1
multi_input[0, 0, :, 2] = 1  # important!
multi_input[1, 1, :, 0:2] = 1
multi_input[1, 0, :, 2] = 1
multi_target = torch.zeros(2, 3, 3, dtype=torch.long)
multi_target[0, :, 1:3] = 2
multi_target[1, :, 1:3] = 1  # important!


def test_pixel_accuracy():
    input = segmentation.binary_as_multiclass(binary_input, 1 / 2)
    assert segmentation.pixel_accuracy(input, binary_target).item() == pytest.approx(3 / 9)
    assert segmentation.pixel_accuracy(multi_input, multi_target).item() == pytest.approx(3 / 9)


def test_classwise_iou():
    assert segmentation.classwise_iou(multi_input, multi_target).tolist() == pytest.approx([0, 1 / 3, 1 / 3])


def test_mean_iou():
    assert segmentation.mean_iou(multi_input, multi_target).item() == pytest.approx(2 / 9)
