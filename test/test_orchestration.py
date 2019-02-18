from tempfile import gettempdir

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from homura import reporter, callbacks, trainer, optim, is_tensorboardX_available


@pytest.mark.parametrize("rep", ["tqdm", "logger", "tensorboard"])
def test(rep):
    tmpdir = str(gettempdir())
    if rep == "tensorboard" and not is_tensorboardX_available:
        pytest.skip("tensorboardX is not available")

    @callbacks.metric_callback_decorator
    def loss(data):
        return data["loss"]

    model = nn.Linear(10, 10)
    optimizer = optim.SGD(lr=0.1)

    c = [callbacks.AccuracyCallback(), loss]
    epoch = range(1)
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]
    with {"tqdm": lambda: reporter.TQDMReporter(epoch, c, tmpdir),
          "logger": lambda: reporter.LoggerReporter(c, tmpdir),
          "tensorboard": lambda: reporter.TensorboardReporter(c, tmpdir)
          }[rep]() as _rep:
        tr = trainer.SupervisedTrainer(model, optimizer, F.cross_entropy,
                                       callbacks=_rep, verb=False)
        if rep == "tqdm":
            epoch = _rep
        for _ in epoch:
            tr.train(loader)
            tr.test(loader)
