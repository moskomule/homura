from pathlib import Path
from tempfile import gettempdir

import pytest
import torch
from homura import reporters, callbacks, optim, is_tensorboardX_available, metrics, trainers
from homura.utils.inferencer import Inferencer
from torch import nn
from torch.nn import functional as F


@pytest.mark.parametrize("rep", ["tqdm", "logger", "tensorboard"])
def test(rep):
    tmpdir = str(gettempdir())
    if rep == "tensorboard" and not is_tensorboardX_available:
        pytest.skip("tensorboardX is not available")

    @callbacks.metric_callback_decorator
    def ca(data):
        output, target = data["output"], data["data"][1]
        return {i: v for i, v in enumerate(metrics.classwise_accuracy(output, target))}

    model = nn.Linear(10, 10)
    optimizer = optim.SGD(lr=0.1)

    c = callbacks.CallbackList(callbacks.AccuracyCallback(), ca, callbacks.WeightSave(tmpdir))
    epoch = range(1)
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]
    with {"tqdm": lambda: reporters.TQDMReporter(epoch, c, tmpdir),
          "logger": lambda: reporters.LoggerReporter(c, tmpdir),
          "tensorboard": lambda: reporters.TensorboardReporter(c, tmpdir)
          }[rep]() as _rep:
        tr = trainers.SupervisedTrainer(model, optimizer, F.cross_entropy,
                                        callbacks=_rep, verb=False)
        if rep == "tqdm":
            epoch = _rep
        for _ in epoch:
            tr.train(loader)
            tr.test(loader)

    save_file = list(Path(tmpdir).glob("*/*.pkl"))[0]
    tr.resume(save_file)

    c = callbacks.AccuracyCallback()
    with {"tqdm": lambda: reporters.TQDMReporter(epoch, c, tmpdir),
          "logger": lambda: reporters.LoggerReporter(c, tmpdir),
          "tensorboard": lambda: reporters.TensorboardReporter(c, tmpdir)
          }[rep]() as _rep:
        inferencer = Inferencer(model, _rep)
        inferencer.load(save_file)
        inferencer.run(loader)
