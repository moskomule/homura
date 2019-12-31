import uuid
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
NOW = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
UNIQUE_ID = str(uuid.uuid4()).split("-")[0]
BASIC_DIR_NAME = NOW + '-' + UNIQUE_ID
DATA = "data"
GPU = "cuda"
CPU = "cpu"
