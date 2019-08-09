# Homura [![CircleCI](https://circleci.com/gh/moskomule/homura/tree/master.svg?style=svg)](https://circleci.com/gh/moskomule/homura/tree/master) [![document](https://img.shields.io/static/v1?label=doc&message=homura&color=blue)](https://moskomule.github.io/homura)

**homura** is a library for fast prototyping DL research.

🔥🔥🔥🔥 *homura* (焰) is *flame* or *blaze* in Japanese. 🔥🔥🔥🔥

## Requirements

### minimal requirements

```
Python>=3.7
PyTorch>=1.2.0
torchvision>=0.4.0
tqdm # automatically installed
tensorboard # automatically installed
```

### optional

```
miniargs
colorlog
```

To enable distributed training using auto mixed precision (AMP), install apex.

```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```

### test

```
pytest .
```

## install

```console
pip install git+https://github.com/moskomule/homura
```

or

```console
git clone https://github.com/moskomule/homura
cd homura; pip install -e .
```


# APIs

## basics

* Device Agnostic
* Useful features

```python
from homura import optim, lr_scheduler
from homura import trainers, callbacks, reporters
from torchvision.models import resnet50
from torch.nn import functional as F

# model will be registered in the trainer
resnet = resnet50()
# optimizer and scheduler will be registered in the trainer, too
optimizer = optim.SGD(lr=0.1, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(milestones=[30,80], gamma=0.1)

# list of callbacks or reporters can be registered in the trainer
with reporters.TensorboardReporter([callbacks.AccuracyCallback(), 
                                    callbacks.LossCallback()]) as reporter:
    trainer = trainers.SupervisedTrainer(resnet, optimizer, loss_f=F.cross_entropy, 
                                         callbacks=reporter, scheduler=scheduler)
```

Now `iteration` of trainer can be updated as follows,

```python
from homura.trainers import TrainerBase, SupervisedTrainer
from homura.utils.containers import Map

def iteration(trainer: TrainerBase, data: Tuple[torch.Tensor]) -> Mapping[torch.Tensor]:
    input, target = data
    output = trainer.model(input)
    loss = trainer.loss_f(output, target)
    results = Map(loss=loss, output=output)
    if trainer.is_train:
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
    # iteration returns at least (loss, output)
    # registered values can be called in callbacks
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

# Note that `iteration` has `user_value`

callbacks.metric_callback_decorator(lambda data: data["user_value"], name='user_value')

# or equivalently,

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

## reproductivity

This method fixes randomness deterministic in its context.

```python
from homura.utils.reproducibility import set_deterministic
with set_deterministic(seed):
    something()
```

## debugging

```python
>>> debug.module_debugger(nn.Sequential(nn.Linear(10, 5), 
                                        nn.Linear(5, 1)), 
                          torch.randn(4, 10))
[homura.debug|2019-02-25 17:57:06|DEBUG] Start forward calculation
[homura.debug|2019-02-25 17:57:06|DEBUG] forward> name=Sequential(1)
[homura.debug|2019-02-25 17:57:06|DEBUG] forward>   name=Linear(2)
[homura.debug|2019-02-25 17:57:06|DEBUG] forward>   name=Linear(3)
[homura.debug|2019-02-25 17:57:06|DEBUG] Start backward calculation
[homura.debug|2019-02-25 17:57:06|DEBUG] backward>   name=Linear(3)
[homura.debug|2019-02-25 17:57:06|DEBUG] backward> name=Sequential(1)
[homura.debug|2019-02-25 17:57:06|DEBUG] backward>   name=Linear(2)
[homura.debug|2019-02-25 17:57:06|INFO] Finish debugging mode
```

# Examples

See [examples](examples).

* [cifar10.py](examples/cifar10.py): training ResNet-20 or WideResNet-28-10 with random crop on CIFAR10
* [imagenet.py](examples/imagenet.py): training a CNN on ImageNet on multi GPUs (single and     multi process)
* [gap.py](examples/gap.py): better implementation of generative adversarial perturbation

For [imagenet.py](examples/imagenet.py), if you want 

* single node single gpu
* single node multi gpus

run `python imagenet.py /path/to/imagenet/root`.

If you want

* single node multi threads multi gpus

run `python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS /path/to/imagenet/root imagenet.py  --distributed`.

If you want

* multi nodes multi threads multi gpus,

run

* `python -m torch.distributed.launch --nnodes=$NUM_NODES --node_rank=0 --master_addr=$MASTER_IP --master_port=$MASTER_PORT --nproc_per_node=$NUM_GPUS imagenet.py /path/to/imagenet/root --distributed` on the master node
* `python -m torch.distributed.launch --nnodes=$NUM_NODES --node_rank=$RANK --master_addr=$MASTER_IP --master_port=$MASTER_PORT --nproc_per_node=$NUM_GPUS imagenet.py /path/to/imagenet/root --distributed` on the other nodes

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