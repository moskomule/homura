import os
from datetime import datetime

# vocabularies
EPOCH = "epoch"
ITER_PER_EPOCH = "iter_per_epoch"
LOSS = "loss"
MODEL = "model"
MODE = "mode"
OPTIMIZER = "optimizer"
SCHEDULER = "scheduler"
OUTPUT = "output"
TARGET = "target"
TEST = "test"
TRAIN = "train"
TRAINER = "trainer"
ITERATION = "iteration"
NOW = datetime.now().strftime("%b%d-%H-%M-%S")
BASIC_DIR_NAME = NOW + f"{os.getpid():0>5}"
DATA = "data"
GPU = "cuda"
CPU = "cpu"
