#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import TrainerKit
from .trainer import MTTrainer
#from .trainer_scaled import MTTrainerScaled
from .trainer_scaled import MTTrainerGradient, MTTrainerFisher, MTTrainerHessian
from .adamsgd import AdamSGD
