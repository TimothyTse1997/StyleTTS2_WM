from pathlib import Path

import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
from scipy.io.wavfile import write

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import phonemizer

from models import *
from utils import *
from text_utils import TextCleaner

from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import (
    DiffusionSampler,
    ADPM2Sampler,
    KarrasSchedule,
    DiffusionNoiseInsertSampler,
    ADPM2NoiseInsersionSampler,
    AEulerDeterministicSampler,
    DiffusionInversionSampler,
    AEulerInverseSampler
)

class LatentInvertor:

    def __init__(self):
        self.device = "cuda"
        self.config_path = "/gpfs/fs3c/nrc/dt/tst000/.cache/huggingface/hub/models--yl4579--StyleTTS2-LibriTTS/snapshots/3aa7ba7f8f275ec13dce21682a61494c35089e2a/Models/LibriTTS/config.yml"

        self.config = yaml.safe_load(open(self.config_path))

        self.model_checkpoint_dir = "/gpfs/fs3c/nrc/dt/tst000/.cache/huggingface/hub/models--yl4579--StyleTTS2-LibriTTS/snapshots/3aa7ba7f8f275ec13dce21682a61494c35089e2a/Models/LibriTTS/"

        self.model, self.model_params = self.load_model(self.config, self.model_checkpoint_dir)

        self.sampler = DiffusionInversionSampler(
            self.model.diffusion.diffusion,
            sampler=AEulerInverseSampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

        pass

    def load_model(self, config, model_checkpoint_dir):
        # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)

        model_params = recursive_munch(config['model_params'])

        model = build_model(model_params, text_aligner, pitch_extractor, plbert)

        _ = [model[key].eval() for key in model]
        _ = [model[key].to(self.device) for key in model]

        params_whole = torch.load(f"{model_checkpoint_dir}epochs_2nd_00020.pth")

        params = params_whole['net']

        for key in model:
            if key in params:
                print('%s loaded' % key)
                try:
                    model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [model[key].eval() for key in model]
        return model, model_params
    
    def oracle_inversion(self, diffusion_steps, list_of_latents):
        pass