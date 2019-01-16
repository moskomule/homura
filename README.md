# Homura

*Homura* is a support tool for research experiments.

*Homura* (焰) is *flame* or *blaze* in Japanese. 🔥🔥🔥🔥

## Requirements

### minimal requirements

```
Python>=3.6
PyTorch>=1.0
torchvision>=0.2.1
tqdm
```

### optional

```
matplotlib
tensorboardX
visdom
miniargs
```

For Distributed training using Synced BN and FP 16, install apex.

```
pip install -U git+https://github.com/NVIDIA/apex.git --install-option="--cuda_ext" --install-option="--cpp_ext"
```

### test

```
pytest
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

## utils

```python
from homura import optim, lr_scheduler
from homura.utils import trainer, callbacks, reporter
from torchvision.models import resnet50
from torch.nn import functional as F

resnet = resnet50()
# model will be registered in the trainer
_optimizer = optim.SGD(lr=0.1, momentum=0.9)
# optimizer will be registered in the trainer
_scheduler = lr_scheduler.MultiStepLR(milestones=[30,80], gamma=0.1)
# list of callbacks
_callbacks = [callbacks.AccuracyCallback(), callbacks.LossCallback()]
# reporter or list of reporters
_reporter = reporter.TensorboardReporter(_callbacks)
_reporter.enable_report_images(keys=["generated", "real"])
_trainer = trainer.SupervisedTrainer(resnet, _optimizer, loss_f=F.cross_entropy, 
                                     callbacks=_reporter, scheduler=_scheduler)
```

Now `iteration` of trainer can be updated as follows,

```python
from homura.utils.containers import Map
def iteration(trainer: Trainer, inputs: Tuple[torch.Tensor]) -> Mapping[torch.Tensor]:
    input, target = trainer.to_device(inputs)
    output = trainer.model(input)
    loss = trainer.loss_f(output, target)
    results = Map(loss=loss, output=output)
    if trainer.is_train:
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
    # registered values can be called in callbacks
    results.user_value = user_value
    return results
   
_trainer.update_iteration(iteration) 
```

Also, `dict` of models, optimizers, loss functions are supported.

```python
_trainer = CustomTrainer({"generator": generator, "discriminator": discriminator},
                         {"generator": gen_opt, "discriminator": dis_opt},
                         {"reconstruction": recon_loss, "generator": gen_loss},
                         **kwargs)
```

## modules

* `homura.modules` contains *attention*, *conditional batchnorm* and *linear backpropagation*.

## vision

* `homura.vision` contains some modules for vision.


# Examples

See [examples](examples).

* [cifar10.py](examples/cifar10.py): training a CNN with random crop on CIFAR10
* [imagenet.py](examples/imagenet.py): training a CNN on ImageNet on multi GPUs (single and     multi process)
* [gap.py](examples/gap.py): better implementation of generative adversarial perturbation