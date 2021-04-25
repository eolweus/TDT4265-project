from yacs.config import CfgNode as CN

cfg = CN()

cfg.MODEL = CN()

cfg.MODEL.INPUT_CHANNELS = 1
cfg.MODEL.NUM_CLASSES = 4

# ---------------------------------------------------------------------------- #
# Backbone  
# ---------------------------------------------------------------------------- #
cfg.MODEL.BACKBONE = CN()

cfg.MODEL.USE_BACKBONE = False

cfg.MODEL.BACKBONE.INPUT_CHANNELS = 1
cfg.MODEL.BACKBONE.OUTPUT_CHANNELS = 128

# ---------------------------------------------------------------------------- #
# Solver - The same as optimizer
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
# train configs
cfg.SOLVER.BATCH_SIZE = 10
cfg.SOLVER.TEST_BATCH_SIZE = 5 
cfg.SOLVER.LEARN_RATE = 0.01

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
cfg.TRAINING.USE_CHECKPOINT = False
cfg.TRAINING.EARLY_STOP_COUNT = 5
cfg.TRAINING.USE_TRANSFORMS = False
cfg.TRAINING.TRANSFORM = "Gaussian_blur"

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------
cfg.TESTING = CN()

cfg.TESTING.CROP_TEE = False
cfg.TESTING.TEST_ONLY = False

# ------------------------------------------------------------

cfg.DATASET = "TTE"

cfg.OUTPUT_DIR = "./outputs/TTE"
cfg.LOG_DIR = "./logs/TTE"
