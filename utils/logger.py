import logging
import os
import sys
from configs import cfg
from pathlib import Path


def setup_logger(name, save_dir=cfg.LOG_DIR):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # save_dir = save_dir+"/log"
    # don't log results for the non-master process
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
