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
PID = f"{os.getpid():0>5}"
BASIC_DIR_NAME = NOW + PID
DATA = "data"
GPU = "cuda"
CPU = "cpu"
