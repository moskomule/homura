from homura.lr_scheduler import multistep_with_warmup


def test_multistep_with_warmup():
    f = multistep_with_warmup(5, [10, 20], 0.1)
    epochs = [0, 1, 4, 5, 10, 20]
    expecteds = [1 / 5, 2 / 5, 1, 1, 0.1 ** 1, 0.1 ** 2]
    for epoch, expected in zip(epochs, expecteds):
        assert f(epoch) == expected
