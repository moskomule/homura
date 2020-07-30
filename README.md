# homura ![](https://github.com/moskomule/homura/workflows/pytest/badge.svg) [![document](https://img.shields.io/static/v1?label=doc&message=homura&color=blue)](https://moskomule.github.io/homura)

**homura** is a library for fast prototyping DL research.

🔥🔥🔥🔥 *homura* (焰) is *flame* or *blaze* in Japanese. 🔥🔥🔥🔥

## Requirements

### Minimal requirements

```
Python>=3.8
PyTorch>=1.6.0
torchvision>=0.7.0
tqdm # automatically installed
tensorboard # automatically installed
hydra-core # automatically installed
```

### Optional

```
colorlog (to log with colors)
faiss (for faster kNN)
accimage (for faster image pre-processing)
horovad (for easier distributed training)
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

### horovod installation

```
conda install gxx_linux-64
pip install horovod
```


# APIs

## Basics

`homura` aims abstract (e.g., device-agnostic) simple prototyping.

```python
from homura import optim, lr_scheduler
from homura import trainers, callbacks, reporters
from torchvision.models import resnet50
from torch.nn import functional as F

# User does not need to care about the device
resnet = resnet50()
# Model is registered in optimizer lazily. This is convenient for distributed training and other complicated scenes.
optimizer = optim.SGD(lr=0.1, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(milestones=[30,80], gamma=0.1)

# `homura` has callbacks
c = [callbacks.AccuracyCallback(),
    reporters.TensorboardReporter(".")]
with trainers.SupervisedTrainer(resnet, optimizer, loss_f=F.cross_entropy, 
                                     callbacks=c, scheduler=scheduler) as trainer:
    # epoch-based training
    for _ in range(epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    # otherwise, iteration-based training

    trainer.run(train_loader, test_loader, 
                total_iterations=1_000, val_intervals=10)
```

User can customize `iteration` of `trainer` as follows.

```python
from homura.trainers import TrainerBase, SupervisedTrainer
from homura.utils.containers import TensorMap

trainer = SupervisedTrainer(...)

def iteration(trainer: TrainerBase, 
              data: Tuple[torch.Tensor]) -> Mapping[torch.Tensor]:
    input, target = data
    output = trainer.model(input)
    loss = trainer.loss_f(output, target)
    results = Map(loss=loss, output=output)
    if trainer.is_train:
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
    # iteration returns at least (loss, output)
    # registered value can be called in callbacks
    results.user_value = user_value
    return results

SupervisedTrainer.iteration = iteration
# or   
trainer.update_iteration(iteration) 
```

`callbacks.Callback` can access the parameters of models, loss, outputs of models and other user-defined values.

In most cases, `callbacks.metric_callback_decorator` is useful. The returned values are accumulated.

```python
from homura import callbacks

@callbacks.metric_callback_decorator
def user_value(data):
    return data["user_value"]
```  

`callbacks.Callback` has methods `before_all`, `before_iteration`, `before_epoch`, `after_all`, `after_iteration` and `after_epoch`. For example, `callbacks.WeightSave` is like:

```python
from homura.callbacks import Callback
class WeightSave(Callback):
    ...

    def after_epoch(self, data: Mapping):
        self._epoch = data["epoch"]
        self._step = data["step"]
        if self.save_freq > 0 and data["epoch"] % self.save_freq == 0:
            self.save(data, f"{data['epoch']}.pkl")

    def after_all(self, data: Mapping):
        if self.save_freq == -1:
            self.save(data, "weight.pkl")
```


`dict` of models, optimizers, loss functions are supported.

```python
trainer = CustomTrainer({"generator": generator, "discriminator": discriminator},
                        {"generator": gen_opt, "discriminator": dis_opt},
                        {"reconstruction": recon_loss, "generator": gen_loss},
                        **kwargs)
```

## Distributed training

Easy distributed initializer `homura.init_distributed()` is available. See [imagenet.py](example/imagenet.py) as an example.


## Reproducibility

This method makes randomness deterministic in its context.

```python
from homura.utils.reproducibility import set_deterministic, set_seed
with set_deterministic(seed):
    something()

with set_seed(seed):
    other_thing()
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

run `python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS imagenet.py distributed.on=true`.

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