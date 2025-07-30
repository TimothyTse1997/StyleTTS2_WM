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
    ADPM2NoiseInsersionSampler
)


class Inferencer:
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

    mean, std = -4, 4

    global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

    textclenaer = TextCleaner()

    def __init__(self):
        self.device = "cuda"
        self.config_path = "/gpfs/fs3c/nrc/dt/tst000/.cache/huggingface/hub/models--yl4579--StyleTTS2-LibriTTS/snapshots/3aa7ba7f8f275ec13dce21682a61494c35089e2a/Models/LibriTTS/config.yml"

        self.config = yaml.safe_load(open(self.config_path))

        self.model_checkpoint_dir = "/gpfs/fs3c/nrc/dt/tst000/.cache/huggingface/hub/models--yl4579--StyleTTS2-LibriTTS/snapshots/3aa7ba7f8f275ec13dce21682a61494c35089e2a/Models/LibriTTS/"

        self.model, self.model_params = self.load_model(self.config, self.model_checkpoint_dir)
        self.sampler = DiffusionNoiseInsertSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2NoiseInsersionSampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

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

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def inference(self, text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, epsilons=None):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self.length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                            num_steps=diffusion_steps,
                                            epsilons=epsilons).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,
                                            s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))


        return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later
    
if __name__ == "__main__":
    from pathlib import Path

    inferencer = Inferencer()

    text = ''' StyleTTS 2 is a text to speech model that leverages style diffusion and adversarial training with large speech language models to achieve human level text to speech synthesis. ''' # @param {type:"string"}
    tts_dataset_path = Path("/gpfs/fs3c/nrc/dt/tst000/LibriTTS/dev-clean/") 
    reference_dicts = {
        speaker_path.name: str(list(speaker_path.glob("**/*.wav"))[0]) for speaker_path in tts_dataset_path.glob("*")
    }
    #reference_dicts['A'] = "/home/tst000/projects/tst000/LibriTTS/dev-clean/174/168635/174_168635_000014_000000.wav"
    #reference_dicts['B'] = "/home/tst000/projects/tst000/LibriTTS/dev-clean/84/121550/84_121550_000007_000000.wav"

    device = "cuda"

    diffusion_steps = 20
    noise = torch.randn(1,1,256).to(device)
    epsilons = [torch.randn_like(noise) for i in range(diffusion_steps-1)]

    special_noise = torch.randn_like(noise)

    for k, path in reference_dicts.items():
        try:
            ref_s = inferencer.compute_style(path)

            wav = inferencer.inference(
                text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=diffusion_steps, embedding_scale=1,
                epsilons=epsilons)

            m = np.max(np.abs(wav))
            wavf32 = (wav/m).astype(np.float32)

            write(f"../_examples/example_{k}_no_aug.wav", 24000, wavf32)
        except: continue

        current_epsilons = epsilons
        for i in range(diffusion_steps-1):
            current_epsilons[i] = special_noise
            start = time.time()
            wav = inferencer.inference(
                text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=diffusion_steps, embedding_scale=1,
                epsilons=current_epsilons)

            m = np.max(np.abs(wav))
            wavf32 = (wav/m).astype(np.float32)

            write(f"../_examples/example_{k}_step_aug_{i}.wav", 24000, wavf32)
