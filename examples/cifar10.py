import torch.nn.functional as F

from homura import optim, lr_scheduler, reporters, callbacks, trainers
from homura.vision.data.loaders import cifar10_loaders
from homura.vision.models.cifar import resnet20, wrn28_10


def main():
    train_loader, test_loader = cifar10_loaders(args.batch_size)

    model = {"resnet20": resnet20,
             "wrn28_10": wrn28_10}[args.model](num_classes=10)
    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR([100, 150])
    c = [callbacks.AccuracyCallback(), callbacks.LossCallback()]

    with reporters.TQDMReporter(range(200), callbacks=c) as tq, reporters.TensorboardReporter(c) as tb:
        trainer = trainers.SupervisedTrainer(model, optimizer, F.cross_entropy, callbacks=[tq, tb],
                                             scheduler=scheduler)
        for _ in tq:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    import miniargs

    p = miniargs.ArgumentParser()
    p.add_int("--batch_size", default=128)
    p.add_str("--model", choices=["resnet20", "wrn28_10"])

    args = p.parse()
    main()
