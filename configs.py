from yacs.config import CfgNode as CN
from decouple import config
TTE_BASE_PATH=config('TTE_BASE_PATH')


cfg = CN()

cfg.MODEL = CN()

# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
cfg.MODEL.NUM_CLASSES = 4


# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
cfg.MODEL.BACKBONE = CN()

cfg.MODEL.BACKBONE.INPUT_CHANNELS = 3
cfg.MODEL.BACKBONE.OUTPUT_CHANNELS = 128

cfg.MODEL

# ---------------------------------------------------------------------------- #
# Solver - The same as optimizer
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
# train configs
cfg.SOLVER.BATCH_SIZE = 12
cfg.SOLVER.LR = 0.01

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
# Image size
cfg.INPUT.IMAGE_SIZE = [300, 300]

# ------------------------------------------------------------

cfg.VISUAL_DEBUG = True
cfg.BATCH_SIZE = 12
cfg.TEST_BATCH_SIZE = 6
cfg.EPOCHS = 50
cfg.LEARN_RATE = 0.01

cfg.TRAIN_PARTITION = 2/3

cfg.DATASET = "TTE"

cfg.OUTPUT_DIR = "./outputs"
# total train data = 450
