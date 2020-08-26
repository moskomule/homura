# homura ![](https://github.com/moskomule/homura/workflows/pytest/badge.svg) [![document](https://img.shields.io/static/v1?label=doc&message=homura&color=blue)](https://moskomule.github.io/homura)

**homura** is a library for fast prototyping DL research.

🔥🔥🔥🔥 *homura* (焰) is *flame* or *blaze* in Japanese. 🔥🔥🔥🔥

## Requirements

### Minimal requirements

```
Python>=3.8
PyTorch>=1.6.0
torchvision>=0.7.0
```

### Optional

```
colorlog (to log with colors)
faiss (for faster kNN)
accimage (for faster image pre-processing)
horovad (for distributed training without using torch.distributed)
cupy
```

### test

```
pytest .
```

## Installation

```console
pip install git+https://github.com/moskomule/homura
```

or

```console
git clone https://github.com/moskomule/homura
cd homura
pip install -e .
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
model = MODEL_REGISTRY('model_name')(...)

# Model is registered in optimizer lazily. This is convenient for distributed training and other complicated scenes.
optimizer = optim.SGD(lr=0.1, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(milestones=[30,80], gamma=0.1)

# from v2020.08, the callbacks system changed
# SupervisedTrainer by default reports loss and accuracy
# TQDMReporter is used as a default reporter.
# If you need additional reporters, do as follows 

with trainers.SupervisedTrainer(model, 
                                optimizer, 
                                F.cross_entropy, 
                                reporters=[reporters.TensorboardReporter(...)],
                                scheduler=scheduler) as trainer:
    # epoch-based training
    for _ in trainer.epoch_iterator(epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    # otherwise, iteration-based training

    trainer.run(train_loader, test_loader, 
                total_iterations=1_000, val_intervals=10)

    print(f"Max Accuracy={max(trainer.history['accuracy']['test'])}")
```

You can customize `iteration` of `trainer` as follows.

```python
from homura.trainers import TrainerBase, SupervisedTrainer

trainer = SupervisedTrainer(...)

# from v2020.08, iteration is much simpler

def iteration(trainer: TrainerBase, 
              data: Tuple[torch.Tensor, torch.Tensor]
              ) -> None:
    input, target = data
    output = trainer.model(input)
    loss = trainer.loss_f(output, target)
    trainer.reporter.add('loss', loss)
    trainer.reporter.add('accuracy', accuracy(input, target))
    if trainer.is_train:
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

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

## Distributed training

Distributed training is complicated at glance. `homura` has simple APIs, to hide the messy codes for DDP, such as `homura.init_distributed` for the initialization and `homura.is_master` for checking if the process is master or not.   

For details, see `examples/imagenet.py`.

## Reproducibility

This method makes randomness deterministic in its context.

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

Following major libraries, `homura` also has a simple register system.

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

* [cifar10.py](examples/cifar10.py): training ResNet-20 or WideResNet-28-10 with random crop on CIFAR10
* [imagenet.py](examples/imagenet.py): training a CNN on ImageNet on multi GPUs (single and multi process)

Note that homura expects datasets are downloaded in `~/.torch/data/DATASET_NAME`.

For [imagenet.py](examples/imagenet.py), if you want 

* single node single gpu
* single node multi gpus

run `python imagenet.py`.

If you want

* single node multi threads multi gpus

run `python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS imagenet.py [...]`.

If you want

* multi nodes multi threads multi gpus,

run

* `python -m torch.distributed.launch --nnodes=$NUM_NODES --node_rank=0 --master_addr=$MASTER_IP --master_port=$MASTER_PORT --nproc_per_node=$NUM_GPUS imagenet.py` on the master node
* `python -m torch.distributed.launch --nnodes=$NUM_NODES --node_rank=$RANK --master_addr=$MASTER_IP --master_port=$MASTER_PORT --nproc_per_node=$NUM_GPUS imagenet.py` on the other nodes

Here, `0<$RANK<$NUM_NODES`.

# Citing

```bibtex
@misc{homura,
    author = {Ryuichiro Hataya},
    title = {homura},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://GitHub.com/moskomule/homura}},
}
```