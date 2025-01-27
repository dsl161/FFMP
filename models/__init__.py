# -*- coding: utf-8 -*-

from .ffmc import FFMC
from .loss.genloss import GeneralizedLoss
from .loss.mseloss import MSELoss
        

def build_model(config):
    model = FFMC(config)
    # return model, MSELoss(config.FACTOR)
    return model, GeneralizedLoss(config.FACTOR)
