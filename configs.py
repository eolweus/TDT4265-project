from yacs.config import CfgNode as CN

cfg = CN()

cfg.VISUAL_DEBUG = True
cfg.BATCH_SIZE = 12
cfg.EPOCHS = 50
cfg.LEARN_RATE = 0.01


# total data = 450
cfg.TEST_SIZE = 70
cfg.VALIDATION_SIZE = 110
