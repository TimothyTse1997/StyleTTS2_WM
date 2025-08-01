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
    AEulerDeterministicSampler
)


class Inferencer:
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

    mean, std = -4, 4

    global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

    textclenaer = TextCleaner()

    def __init__(self, diffusion_class=DiffusionNoiseInsertSampler, sampler_class=ADPM2NoiseInsersionSampler):
        self.device = "cuda"
        self.config_path = "/gpfs/fs3c/nrc/dt/tst000/.cache/huggingface/hub/models--yl4579--StyleTTS2-LibriTTS/snapshots/3aa7ba7f8f275ec13dce21682a61494c35089e2a/Models/LibriTTS/config.yml"

        self.config = yaml.safe_load(open(self.config_path))

        self.model_checkpoint_dir = "/gpfs/fs3c/nrc/dt/tst000/.cache/huggingface/hub/models--yl4579--StyleTTS2-LibriTTS/snapshots/3aa7ba7f8f275ec13dce21682a61494c35089e2a/Models/LibriTTS/"

        self.model, self.model_params = self.load_model(self.config, self.model_checkpoint_dir)
        self.sampler = diffusion_class(
            self.model.diffusion.diffusion,
            sampler=sampler_class(),
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

    def inference(
        self, 
        noise, text, ref_s, alpha = 0.3, beta = 0.7,
        diffusion_steps=5, embedding_scale=1,
        epsilons=None, log_intermediate_steps=False
    ):
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
            
            result = self.sampler(noise = noise.to(self.device),
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                            num_steps=diffusion_steps,
                                            epsilons=epsilons,
                                            return_intermediate=log_intermediate_steps)
            if not log_intermediate_steps:
                s_pred = result.squeeze(1)
            else:
                s_pred, x_steps = result
                s_pred = s_pred.squeeze(1)

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

        out = out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

        if not log_intermediate_steps:
            return out

        return out, x_steps

def generate_sample_LibriTTS(
    text = "I go to school by bus.",
    tts_dataset_path=Path("/gpfs/fs3c/nrc/dt/tst000/LibriTTS/dev-clean/"),
    output_dir="../_examples"
):
    #text = ''' StyleTTS 2 is a text to speech model that leverages style diffusion and adversarial training with large speech language models to achieve human level text to speech synthesis. ''' # @param {type:"string"}
    inferencer = Inferencer()
    reference_dicts = {
        speaker_path.name: str(list(speaker_path.glob("**/*.wav"))[0]) for speaker_path in tts_dataset_path.glob("*")
    }
    device = "cuda"

    diffusion_steps = 20
    noise = torch.randn(1,1,256).to(device)
    #epsilons = [torch.randn_like(noise) for i in range(diffusion_steps-1)]
    epsilons = [torch.zeros_like(noise) for i in range(diffusion_steps-1)]

    special_noise = torch.randn_like(noise)

    for k, path in reference_dicts.items():
        try:
        #if True:
            ref_s = inferencer.compute_style(path)

            wav = inferencer.inference(
                noise, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=diffusion_steps, embedding_scale=1,
                epsilons=epsilons)

            m = np.max(np.abs(wav))
            wavf32 = (wav/m).astype(np.float32)

            write(f"{output_dir}/example_{k}_no_aug.wav", 24000, wavf32)
        except: 
            continue

        current_epsilons = epsilons
        for i in range(diffusion_steps-1):
            current_epsilons[i] = special_noise
            start = time.time()
            wav = inferencer.inference(
                noise, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=diffusion_steps, embedding_scale=1,
                epsilons=current_epsilons)

            m = np.max(np.abs(wav))
            wavf32 = (wav/m).astype(np.float32)

            write(f"{output_dir}/example_{k}_step_aug_{i}.wav", 24000, wavf32)

def generate_audio_with_intermediate_steps(
    text = "I go to school by bus.",
    tts_dataset_path=Path("/gpfs/fs3c/nrc/dt/tst000/LibriTTS/dev-clean/"),
    output_dir=Path("/home/tst000/projects/tst000/intermidiate_diffusion_steps_styletts2_no_noise_euler/"),
    num_max_speaker=5,
    diffusion_steps=20,
):

    inferencer = Inferencer(
        diffusion_class=DiffusionNoiseInsertSampler,
        sampler_class=AEulerDeterministicSampler
    )

    reference_dicts = {
        speaker_path.name: str(list(speaker_path.glob("**/*.wav"))[0]) for speaker_path in tts_dataset_path.glob("*")
    }
    device = "cuda"

    noise = torch.randn(1,1,256).to(device)

    for num, (k, path) in enumerate(reference_dicts.items()):
        try:
        #if True:
            ref_s = inferencer.compute_style(path)

            wav, x_steps = inferencer.inference(
                noise, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=diffusion_steps, embedding_scale=1,
                log_intermediate_steps=True
            )

            m = np.max(np.abs(wav))
            wavf32 = (wav/m).astype(np.float32)
            save_dir = output_dir /  "audios"
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            write(save_dir / f"example_{k}_no_aug.wav", 24000, wavf32)

            torch.save(
                ref_s.detach().cpu(), save_dir / f"{k}_ref_s.pt")
            with open(save_dir / f"{k}_text.txt", 'w') as f:
                f.write(text)

            save_dir = output_dir / "latents" / k
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            torch.save(
                noise, save_dir / f"0.pt")

            for i, x_t in enumerate(x_steps):
                x_t = x_t.detach().cpu()
                torch.save(
                    x_t, save_dir / f"{i+1}.pt")
            
            print(k)
            if num>num_max_speaker: break
        except:
            continue

    pass

if __name__ == "__main__":
    generate_sample_LibriTTS()
    #generate_audio_with_intermediate_steps()
    pass

    