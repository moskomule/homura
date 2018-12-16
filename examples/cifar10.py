import torch.nn.functional as F
from homura import optim, lr_scheduler
from homura.utils import reporter, callbacks, Trainer
from homura.vision.data.loaders import cifar10_loaders
from homura.vision.models.cifar import resnet20


def main(batch_size):
    train_loader, test_loader = cifar10_loaders(batch_size)

    model = resnet20(num_classes=10)
    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR([100, 150])
    c = [callbacks.AccuracyCallback(), callbacks.LossCallback()]
    r = reporter.TQDMReporter(range(200), callbacks=c)
    tb = reporter.TensorboardReporter(c)
    tb.report_parameters()

    with callbacks.CallbackList(r, tb) as rep:
        trainer = Trainer(model, optimizer, F.cross_entropy, callbacks=rep, scheduler=scheduler)
        for _ in r:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    main(128)
