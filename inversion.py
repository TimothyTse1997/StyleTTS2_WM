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

from inference import Inferencer

class LatentInvertor(Inferencer):

    def __init__(self, inner_loop=5):
        self.device = "cuda"
        self.config_path = "/gpfs/fs3c/nrc/dt/tst000/.cache/huggingface/hub/models--yl4579--StyleTTS2-LibriTTS/snapshots/3aa7ba7f8f275ec13dce21682a61494c35089e2a/Models/LibriTTS/config.yml"

        self.config = yaml.safe_load(open(self.config_path))

        self.model_checkpoint_dir = "/gpfs/fs3c/nrc/dt/tst000/.cache/huggingface/hub/models--yl4579--StyleTTS2-LibriTTS/snapshots/3aa7ba7f8f275ec13dce21682a61494c35089e2a/Models/LibriTTS/"

        self.model, self.model_params = self.load_model(self.config, self.model_checkpoint_dir)

        self.sampler = DiffusionInversionSampler(
            self.model.diffusion.diffusion,
            sampler=AEulerInverseSampler(inner_loop=inner_loop),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

        pass

    @torch.no_grad()   
    def load_wav_to_style(self, wavfname, ref_s, alpha = 0.3, beta = 0.7,):
        recon_ref = self.compute_style(wavfname)
        recon_s, recon_ref = recon_ref[:, 128:], recon_ref[:, :128]

        recon_ref = (recon_ref - ref_s[:, :128] * (1-alpha)) / alpha
        recon_s = (recon_s - (1-beta) * ref_s[:, 128:]) / beta

        return torch.cat([recon_s, recon_ref], dim=1)

    @torch.no_grad()   
    def oracle_inversion(
        self, text, ref_s, list_of_latents, diffusion_steps=20, 
        alpha = 0.3, beta = 0.7, embedding_scale=1
    ):
        intermiate_steps = list_of_latents[1:]

        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = self.length_to_mask(input_lengths).to(self.device)

        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        inv_noise, inv_steps = self.sampler(
            audio=None,
            num_steps=diffusion_steps,
            #return_intermediate=log_intermediate_steps)
            oracle_steps=intermiate_steps,
            #epsilons=epsilons
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s, # reference from the same speaker as the embedding
        )

        return inv_noise, inv_steps
    

def inversion_test(
    data_dir='/home/tst000/projects/tst000/intermidiate_diffusion_steps_styletts2_no_noise_euler',
    device="cuda",
    invertor=None,#LatentInvertor()
):
    data_dir = Path(data_dir)
    
    for speaker_latent_dir in (data_dir / "latents").glob("*"):
        speaker = speaker_latent_dir.name

        ref_s = torch.load(data_dir / "audios" / f"{speaker}_ref_s.pt").to(device)
        text = open(data_dir / "audios" / f"{speaker}_text.txt", 'r').readline()

        latents = []
        for i in range(20):
            fname = f"{i}.pt"
            latent = torch.load(speaker_latent_dir / fname).to(device)
            latents.append(latent)
        
        inv_noise, inv_steps = invertor.oracle_inversion(
            text, ref_s, latents
        )
        print(inv_noise.shape)
        print(inv_steps[0].shape)
        print(len(inv_steps))

        for i in range(19):
            print(i, ((inv_steps[i] - latents[i+1]) ** 2).mean())
        break
    pass

if __name__ == "__main__":
    invertor = LatentInvertor()
    for i in range(1, 10):
        print(invertor.sampler.sampler)
        invertor.sampler.sampler.inner_loop = i
        invertor.sampler.sampler.latent_average = True
        print(f"=================inner loop: {i}=================")
        inversion_test(invertor=invertor)