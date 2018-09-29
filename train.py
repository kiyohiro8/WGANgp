# -*- coding: utf-8 -*-

from config import Config
from WGANgp import  WGANgp

if __name__ == "__main__":
    config = Config()
    model = WGANgp(config)
    if config.RESUME_TRAIN:
        model.resume_train()
    else:
        model.train()