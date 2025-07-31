import re
from tqdm import tqdm

from scipy.io import wavfile
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms

from datasets import load_dataset, Audio


def pad_to_shape(max_shape, codec):
    if codec.shape[-1] == max_shape:
        return codec

    num_pad = max_shape - codec.shape[-1]
    left_pad = num_pad // 2
    right_pad = num_pad - left_pad
    
    padded_codec = F.pad(codec, (left_pad, right_pad), "constant", 0)
    padded_codec[:, :, :left_pad] = codec[:, :, :1]
    padded_codec[:, :, -right_pad:] = codec[:, :, -1:]

    return padded_codec

def wav2mel(fname):
    waveform, sample_rate = torchaudio.load(fname, normalize=True)
    transform = transforms.MelSpectrogram(sample_rate)
    mel_specgram = transform(waveform)
    #print(mel_specgram.shape)
    return mel_specgram # mel_specgram

@torch.no_grad()
def mel_L1(melA, melB):
    loss = nn.L1Loss()
    output = loss(melB, melA)
    return output

def encodec_test(audio_dir):
    import re
    from pathlib import Path
    from collections import defaultdict

    get_speaker = lambda name: re.findall(r"example\_(\d+)\_", name)[0]
    get_step = lambda name: int(re.findall(r"aug\_(\d+)\.wav", name)[0])

    audio_dir = Path(audio_dir)
    no_aug_dict = {}
    aug_dict = defaultdict(dict)

    for p in audio_dir.glob("example_*_no_aug*"):
        speaker = get_speaker(p.name)
        
        mel = wav2mel(p)
        no_aug_dict[speaker] = mel

    print("complete encode no augmentation audios")
    
    for p in audio_dir.glob("example_*_step_aug*"):
        speaker = get_speaker(p.name)
        steps = get_step(p.name)

        mel = wav2mel(p)
        aug_dict[speaker][steps] = mel
    
    print("complete encode augmentated audios")

    all_speakers = list(no_aug_dict.keys())
    steps_L1_dict = defaultdict(list)

    for speaker in all_speakers:
        speaker_aug_dict = aug_dict[speaker]
        step_index = list(speaker_aug_dict.keys())

        source_embed = no_aug_dict[speaker]
        aug_embeds = list(speaker_aug_dict.values())

        max_shape = source_embed.shape[-1]
        for e in aug_embeds:
            max_shape = max(max_shape, e.shape[-1])

        source_embed = pad_to_shape(max_shape, source_embed)
        aug_embeds = [pad_to_shape(max_shape, a) for a in aug_embeds]

        l1_diff = [mel_L1(source_embed, m) for m in aug_embeds] #batch_cosine_similarity(source_embed, aug_embeds)
        for index, score in zip(step_index, l1_diff):
            steps_L1_dict[int(index)].append(score)
    
    print("complete audios L1")
    return steps_L1_dict

if __name__ == "__main__":
    from pathlib import Path
    #wav2mel("/home/tst000/projects/tst000/_examples/example_1673_no_aug.wav")#fname)
    test_dir = "../_examples/"
    results = encodec_test(test_dir)
    sorted_keys = sorted(list(results.keys()))
    print(sorted_keys)
    v = [np.mean(results[k]) for k in sorted_keys]
    print(v)