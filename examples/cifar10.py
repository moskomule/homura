import torch.nn.functional as F

from homura import optim, lr_scheduler
from homura.utils import reporter, callbacks, Trainer
from homura.vision.data.loaders import cifar10_loaders
from homura.vision.models.cifar import resnet20


def main():
    train_loader, test_loader = cifar10_loaders(args.batch_size)

    model = resnet20(num_classes=10)
    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR([100, 150])
    c = [callbacks.AccuracyCallback(), callbacks.LossCallback()]
    r = []
    if args.use_tqdm:
        r.append(reporter.TQDMReporter(range(200), callbacks=c))
    if args.use_tb:
        r.append(reporter.TensorboardReporter(c))
        r[-1].report_params(model)

    rep = None if len(r) == 0 else callbacks.CallbackList(*r)
    trainer = Trainer(model, optimizer, F.cross_entropy, callbacks=rep, scheduler=scheduler)
    it = range(200) if not args.use_tqdm else r[0]
    for _ in it:
        trainer.train(train_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--use_tb", action="store_true")
    p.add_argument("--use_tqdm", action="store_true")

    args = p.parse_args()
    main()
