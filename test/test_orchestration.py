from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from homura import reporters, callbacks, optim, metrics, trainers
from homura.utils.inferencer import Inferencer


@pytest.mark.parametrize("rep", ["tqdm", "logger", "tensorboard"])
@pytest.mark.parametrize("save_freq", [-1, 1])
def test(tmp_path, rep, save_freq):
    temp_dir = tmp_path / "test"

    @callbacks.metric_callback_decorator
    def ca(data):
        output, target = data["output"], data["data"][1]
        return {i: v for i, v in enumerate(metrics.classwise_accuracy(output, target))}

    model = nn.Linear(10, 10)
    optimizer = optim.SGD(lr=0.1)

    c = callbacks.CallbackList(callbacks.AccuracyCallback(), ca, callbacks.WeightSave(save_path=temp_dir,
                                                                                      save_freq=save_freq))
    epoch = range(1)
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]
    with {"tqdm": lambda: reporters.TQDMReporter(epoch, c, temp_dir),
          "logger": lambda: reporters.LoggerReporter(c, temp_dir),
          "tensorboard": lambda: reporters.TensorboardReporter(c, temp_dir)
          }[rep]() as _rep:
        tr = trainers.SupervisedTrainer(model, optimizer, F.cross_entropy,
                                        callbacks=_rep, verb=False)
        if rep == "tqdm":
            epoch = _rep
        for _ in epoch:
            tr.train(loader)
            tr.test(loader)
        tr.exit()

    try:
        # .../test/**/0.pkl
        save_file = list(Path(temp_dir).glob("*/*.pkl"))[0]
    except IndexError as e:
        print(list(Path(temp_dir).glob("*/*")))
        raise e
    tr.resume(save_file)

    c = callbacks.AccuracyCallback()
    with {"tqdm": lambda: reporters.TQDMReporter(epoch, c, temp_dir),
          "logger": lambda: reporters.LoggerReporter(c, temp_dir),
          "tensorboard": lambda: reporters.TensorboardReporter(c, temp_dir)
          }[rep]() as _rep:
        inferencer = Inferencer(model, _rep)
        inferencer.load(save_file)
        inferencer.run(loader)
