# homura  [![document](https://img.shields.io/static/v1?label=doc&message=homura&color=blue)](https://moskomule.github.io/homura)

| master | dev |
| --- | --- |
| ![pytest](https://github.com/moskomule/homura/workflows/pytest/badge.svg) | ![pytest](https://github.com/moskomule/homura/workflows/pytest/badge.svg?branch=dev)  |

**homura** is a fast prototyping library for DL research.

🔥🔥🔥🔥 *homura* (焰) is *flame* or *blaze* in Japanese. 🔥🔥🔥🔥

## Requirements

### Minimal requirements

```
Python>=3.9
PyTorch>=1.9.0
torchvision>=0.10.0
```

## Installation

```console
pip install -U homura-core
```

or

```console
pip uninstall homura
pip install -U git+https://github.com/moskomule/homura
```

## Optional

```
faiss (for faster kNN)
accimage (for faster image pre-processing)
cupy (homura)
opt_einsum (for accelerated einsum)
```

## test

```
pytest .
```

# APIs

## Basics

`homura` aims abstract (e.g., device-agnostic) simple prototyping.

```python
from homura import optim, lr_scheduler
from homura import trainers, reporters
from homura.vision import MODEL_REGISTRY, DATASET_REGISTRY
from torch.nn import functional as F

train_loader, test_loader, num_classes = DATASET_REGISTRY('dataset_name')(...)
# User does not need to care about the device
model = MODEL_REGISTRY('model_name')(num_classes=num_classes)

# Model is registered in optimizer lazily. This is convenient for distributed training and other complicated scenes.
optimizer = optim.SGD(lr=0.1, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(milestones=[30, 80], gamma=0.1)

with trainers.SupervisedTrainer(model,
                                optimizer,
                                F.cross_entropy,
                                reporters=[reporters.TensorboardReporter(...)],
                                scheduler=scheduler) as trainer:
    # epoch-based training
    for _ in trainer.epoch_iterator(num_epochs):
        trainer.train(train_loader)
        trainer.scheduler.step()
        trainer.test(test_loader)
        trainer.scheduler.step()

    # otherwise, iteration-based training

    trainer.run(train_loader, test_loader,
                total_iterations=1_000, val_intervals=10)

    print(f"Max Accuracy={max(trainer.history['accuracy']['tests'])}")
```

You can customize `iteration` of `trainer` as follows.

```python
from homura.trainers import TrainerBase, SupervisedTrainer
from homura.metrics import accuracy

trainer = SupervisedTrainer(...)


# from v2020.08, iteration is much simpler

def iteration(trainer: TrainerBase,
              data: Tuple[torch.Tensor, torch.Tensor]
              ) -> None:
    input, target = data
    output = trainer.model(input)
    loss = trainer.loss_f(output, target)
    trainer.reporter.add('loss', loss.detach())
    trainer.reporter.add('accuracy', accuracy(input, target))
    trainer.reporter.add('')
    if trainer.is_train:
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        # in case schedule is step-wise
        trainer.scheduler.step()


SupervisedTrainer.iteration = iteration
# or   
trainer.update_iteration(iteration) 
```

`dict` of models, optimizers, loss functions are supported. This is useful for GANs, for example.

```python
trainer = CustomTrainer({"generator": generator, "discriminator": discriminator},
                        {"generator": gen_opt, "discriminator": dis_opt},
                        {"reconstruction": recon_loss, "generator": gen_loss},
                        **kwargs)
```

`reporter` internally tracks the values during each epoch and reduces after every epoch. Therefore, users can compute
mIoU, for example, as

```python
from homura.metrics import confusion_matrix


def cm_to_miou(cms: List[torch.Tensor]) -> torch.Tensor:
    # cms: list of confusion matrices
    cm = sum(cms).float()
    miou = cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag())
    return miou.mean().item()


def iteration(trainer: TrainerBase,
              data: Tuple[torch.Tensor, torch.Tensor]
              ) -> None:
    input, target = data
    output = trainer.model(input)
    trainer.reporter.add('miou', confusion_matrix(output, target), reduction=cm_to_miou)
    ...
```

## Distributed training

Distributed training is complicated at glance. `homura` has simple APIs, to hide the messy codes for DDP, such
as `homura.init_distributed` for the initialization and `homura.is_master` for checking if the process is master or not.

For details, see `examples/imagenet.py`.

## Reproducibility

These methods make randomness deterministic in its context.

```python
from homura.utils.reproducibility import set_deterministic, set_seed

with set_deterministic(seed):
    # suppress nondeterministic computation
    # but will affect the performance
    something()

with set_seed(seed):
    # only set random seed of Python, PyTorch and Numpy
    other_thing()
```

## Registry System

Following major libraries, `homura` also has a simple registry system.

```python
from homura import Registry

MODEL_REGISTRY = Registry("language_models")


@MODEL_REGISTRY.register
class Transformer(nn.Module):
    ...


# or

MODEL_REGISTRY.register(bert_model, 'bert')

# magic
MODEL_REGISTRY.import_modules(".")

transformer = MODEL_REGISTRY('Transformer')(...)
# or
bert = MODEL_REGISTRY('bert', ...)
```

# Examples

See [examples](examples).

# Citing

```bibtex
@misc{homura,
    author = {Ryuichiro Hataya},
    title = {homura},
    year = {2018},
    howpublished = {\url{https:/github.com/moskomule/homura}},
}
```