import pytest
import torch

from homura.reporters import _ReporterBase, ReporterList


def test_reporters_list():
    # _ReporterBase is a dummy
    # Not a perfect test
    reporter_list = ReporterList([_ReporterBase()])
    values = torch.as_tensor([5, 20, 30, 1, 4, 8], dtype=torch.float)
    batches = torch.as_tensor([32, 32, 32, 32, 32, 10])
    for value, batch in zip(values, batches):
        reporter_list.set_batch_size(batch.item())
        reporter_list.add("loss", value)
    reporter_list.report()
    assert pytest.approx(reporter_list.history["loss"] == (values * batches).mean().item())

    reporter_list.exit()
    assert len(reporter_list.history) == 0
