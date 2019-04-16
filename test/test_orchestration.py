from pathlib import Path
from tempfile import gettempdir

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from homura import reporters, callbacks, optim, is_tensorboardX_available, metrics, trainers
from homura.utils.inferencer import Inferencer


@pytest.mark.parametrize("rep", ["tqdm", "logger", "tensorboard"])
@pytest.mark.parametrize("save_freq", [-1, 1])
def test(rep, save_freq):
    tmpdir = str(gettempdir())
    if rep == "tensorboard" and not is_tensorboardX_available:
        pytest.skip("tensorboardX is not available")

    @callbacks.metric_callback_decorator
    def ca(data):
        output, target = data["output"], data["data"][1]
        return {i: v for i, v in enumerate(metrics.classwise_accuracy(output, target))}

    model = nn.Linear(10, 10)
    optimizer = optim.SGD(lr=0.1)

    c = callbacks.CallbackList(callbacks.AccuracyCallback(), ca, callbacks.WeightSave(tmpdir, save_freq=save_freq))
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

    # circle-CI cause an error
    save_files = list(Path(tmpdir).glob("*/*.pkl"))
    try:
        save_file = save_files[0]
    except IndexError as e:
        print(save_files)
        raise e
    tr.resume(save_file)

    c = callbacks.AccuracyCallback()
    with {"tqdm": lambda: reporters.TQDMReporter(epoch, c, tmpdir),
          "logger": lambda: reporters.LoggerReporter(c, tmpdir),
          "tensorboard": lambda: reporters.TensorboardReporter(c, tmpdir)
          }[rep]() as _rep:
        inferencer = Inferencer(model, _rep)
        inferencer.load(save_file)
        inferencer.run(loader)
