import torch
from homura.vision.metrics import segmentation_metrics
from pytest import approx


def test_segmentation_metrics():
    output = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 4, 4)
    output[0, 0, 2:4, :] = 1
    output[0, 1, 0:2, :] = 1
    target[0, :, 0:3] = 1
    # output target
    # 1100   1111
    # 1100   1111
    # 1100   1111
    # 1100   0000
    metrics = segmentation_metrics(output, target.long())
    class_iou = metrics["class_iou"]
    assert class_iou[0].item() == approx(1 / 5)
    assert class_iou[1].item() == approx(3 / 7)
    assert metrics["mean_iou"].item() == approx(11 / 35)
