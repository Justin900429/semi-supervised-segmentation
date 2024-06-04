from colorama import Fore, Style
from tabulate import tabulate
from yacs.config import CfgNode as CN


def create_cfg():
    cfg = CN()
    cfg.PROJECT_DIR = None

    # ==========  Model   ==========
    cfg.MODEL = CN()
    cfg.MODEL.IN_CHANNELS = 3
    cfg.MODEL.CHECKPOINT = None

    # ========== Data ==========
    cfg.DATA = CN()
    cfg.DATA.ROOT = None
    cfg.DATA.IMG_SIZE = 512
    cfg.DATA.BATCH_SIZE = 4
    cfg.DATA.NUM_WORKERS = 4
    cfg.DATA.NUM_CLASSES = 10

    # ========== Training ==========
    cfg.TRAIN = CN()
    cfg.TRAIN.USE_SSL = True
    cfg.TRAIN.EPOCHS = 100
    cfg.TRAIN.LOG_EVERY_STEP = 10
    cfg.TRAIN.VAL_FREQ = 10
    cfg.TRAIN.LR = 3e-1
    cfg.TRAIN.WEIGHT_DECAY = 5e-5
    cfg.TRAIN.GRAD_CLIP = 1.0
    cfg.TRAIN.THRESHOLD = 0.95
    cfg.TRAIN.TEMPERATURE = 1.0
    cfg.TRAIN.UNLABEL_WEIGHT = 0.3
    cfg.TRAIN.SAVE_BEST_ONLY = True

    # ========== EMA ==============
    cfg.TRAIN.EMA = CN()
    cfg.TRAIN.EMA.BETA = 0.999
    cfg.TRAIN.EMA.UPDATE_AFTER_STEP = 1
    cfg.TRAIN.EMA.UPDATE_EVERY = 1

    return cfg


def show_config(cfg):
    table = tabulate(
        {"Configuration": [str(cfg)]}, headers="keys", tablefmt="fancy_grid"
    )
    print(f"{Fore.BLUE}", end="")
    print(table)
    print(f"{Style.RESET_ALL}", end="")
