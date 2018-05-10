import torch
from ...utils.exceptions import ShapeError


def _iou(storage: torch.Tensor) -> torch.Tensor:
    # storage: BxNxN
    den = torch.stack([t.diag() for t in storage])
    nom = storage.sum(dim=1) + storage.sum(dim=2) - den
    # return BxN
    return den / nom


def segmentation_metrics(output: torch.Tensor, target: torch.Tensor) -> dict:
    """
    calculate semantic metrics
    :param output: Output from network, BxNxHxW
    :param target: Corresponding target, BxHxW
    :return: dict
    """
    batch_size, num_classes, o_h, o_w = output.size()
    if target.shape != (batch_size, o_h, o_w):
        raise ShapeError(f"output: {output.shape} but target: {output.shape}")

    pred = output.argmax(dim=1).view(batch_size, -1)
    target = target.view(batch_size, -1)
    ones = torch.ones_like(pred)
    storage = torch.zeros(batch_size, num_classes, num_classes)
    for i in range(num_classes):
        for j in range(num_classes):
            storage[:, i, j] = (((i * ones) == target) * ((j * ones) == pred)).float().sum(dim=1)
    class_iou = _iou(storage).mean(dim=0)
    mean_iou = class_iou.mean(dim=0)

    return {"class_iou": class_iou,
            "mean_iou": mean_iou}
