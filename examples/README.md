# Examples

## Requirements

* `homura` by `pip install -U homura-core`
* `chika` by `pip install -U chika`

## Contents

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

* `python -m torch.distributed.launch --nnodes=$NUM_NODES --node_rank=0 --master_addr=$MASTER_IP --master_port=$MASTER_PORT --nproc_per_node=$NUM_GPUS imagenet.py`
  on the master node
* `python -m torch.distributed.launch --nnodes=$NUM_NODES --node_rank=$RANK --master_addr=$MASTER_IP --master_port=$MASTER_PORT --nproc_per_node=$NUM_GPUS imagenet.py`
  on the other nodes

Here, `0<$RANK<$NUM_NODES`.