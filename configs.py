from yacs.config import CfgNode as CN
from decouple import config


cfg = CN()

cfg.MODEL = CN()

# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
cfg.MODEL.NUM_CLASSES = 4
cfg.MODEL.INPUT_CHANNELS = 1


# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
cfg.MODEL.BACKBONE = CN()

cfg.MODEL.BACKBONE.INPUT_CHANNELS = 1
cfg.MODEL.BACKBONE.OUTPUT_CHANNELS = 128

# ---------------------------------------------------------------------------- #
# Solver - The same as optimizer
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
# train configs
cfg.SOLVER.BATCH_SIZE = 12
cfg.SOLVER.TEST_BATCH_SIZE = 6
cfg.SOLVER.LR = 0.01

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
# Image size
cfg.INPUT.IMAGE_SIZE = (384, 384)

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------
cfg.TRAINING = CN()

cfg.TRAINING.VISUAL_DEBUG = True
cfg.TRAINING.TRAIN_PARTITION = 2/3
cfg.TRAINING.EPOCHS = 50

# ------------------------------------------------------------

cfg.DATASET = "TTE"

cfg.OUTPUT_DIR = "./outputs"
cfg.LOG_DIR = "./logs"
