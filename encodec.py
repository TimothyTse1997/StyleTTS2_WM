import re
from tqdm import tqdm

from scipy.io import wavfile
import numpy as np

import torch
import torch.nn.functional as F

from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor, EncodecFeatureExtractor

class EncodecAugModule:

    def __init__(self,
        model_name="facebook/encodec_24khz",
        device="cuda",
    ):
        # load the model + processor (for pre-processing the audio)
        self.model_name = model_name
        self.device = device
        self.model = EncodecModel.from_pretrained(model_name)
        _ = self.model.to(self.device)
        _ = self.model.eval()
        #self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor = EncodecFeatureExtractor.from_pretrained(model_name)
        #self.feature_extractor = EncodecFeatureExtractor.from_pretrained(model_name)
    
    def load_inputs_from_file(self, wavfname):
        arr = wavfile.read(wavfname)
        arr = np.array(arr[1],dtype=float)
        inputs = self.processor(
            raw_audio=arr, 
            sampling_rate=self.processor.sampling_rate, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        #for k, v in inputs.items():
        #    _ = v.to(self.device)
        return inputs

    @torch.no_grad() 
    def encode(self, inputs):
        encoder_outputs = self.model.encode(inputs["input_values"], inputs["padding_mask"])
        return encoder_outputs
    
    def encode_to_codec(self, inputs):
        encoder_outputs = self.encode(inputs)
        return encoder_outputs["audio_codes"]
    
    @torch.no_grad() 
    def decode(self, inputs, encoder_outputs):
        audio_values = self.model.decode(
            encoder_outputs["audio_codes"],
            encoder_outputs["audio_scales"],
            inputs["padding_mask"]
        )[0]
        return audio_values
    
    @torch.no_grad() 
    def recon(self, inputs):
        audio_values = self.model(inputs["input_values"], inputs["padding_mask"]).audio_values

        return audio_values # batch_size, channel, audio arr
    
    @torch.no_grad() 
    def get_embed_from_codec(self, audio_codes):
        results = []
        for frame in audio_codes:
            embeds = self.model.quantizer.decode(
                frame.transpose(0, 1)
            )
            results.append(embeds)
        return torch.stack(results)
            

def _cosine_similarity(A, B):
    """
    Computes the average cosine similarity between a vector A and each row in matrix B.

    Parameters:
    - A (np.ndarray): A 1D numpy array of shape (d,).
    - B (np.ndarray): A 2D numpy array of shape (10, d), where each row is a vector.

    Returns:
    - float: The average cosine similarity between A and the vectors in B.
    """
    # Ensure A is a 1D vector and B is a 2D matrix
    A = A.reshape(1, -1)  # shape: (1, d)
    B = B.reshape(-1, A.shape[1])  # shape: (10, d)

    # Normalize A and B
    A_norm = A / np.linalg.norm(A)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)

    # Compute cosine similarities (dot product between A and each row in B)
    similarities = np.dot(B_norm, A_norm.T).flatten()

    # Return
    return similarities

def _count_same_element(A, B):
    return (A == B).sum(1)/A.shape[-1]

def tensor_to_numpy(target_tensor):
    return target_tensor.detach().cpu().numpy()

def batch_cosine_similarity(source_audio_codec, aug_audio_codec):
    source_audio_codec = source_audio_codec.flatten(2, 3).squeeze(0).squeeze(0)
    aug_audio_codec = aug_audio_codec.flatten(2, 3).squeeze(1)
    assert(source_audio_codec.shape[0] == aug_audio_codec.shape[-1])

    source_audio_codec = tensor_to_numpy(source_audio_codec)
    aug_audio_codec = tensor_to_numpy(aug_audio_codec)

    #return _count_same_element(source_audio_codec, aug_audio_codec) #_cosine_similarity(source_audio_codec, aug_audio_codec)
    return _cosine_similarity(source_audio_codec, aug_audio_codec) #_cosine_similarity(source_audio_codec, aug_audio_codec)


def debug():
    import torchaudio
    encodec = EncodecAugModule()
    test_fname = "/home/tst000/projects/tst000/_examples/example_1673_no_aug.wav"
    inputs = encodec.load_inputs_from_file(test_fname)
    encode = encodec.encode(inputs)
    codec = encodec.encode_to_codec(inputs)
    recon_debug = encodec.decode(inputs, encode)
    print(recon_debug.shape)
    #recon_forward = encodec.recon(inputs)

    #print(codec.shape)
    #print(recon_debug.shape)
    #print(recon_forward.shape)
    #dummy_codec = codec.repeat(4, 1, 1, 1)
    #print(dummy_codec.shape)
    #print(batch_cosine_similarity(codec, dummy_codec))
    #print(encodec.get_embed_from_codec(codec).shape)
    torchaudio.save("/home/tst000/projects/tst000/_examples/debug.wav", recon_debug.detach().cpu()[0], sample_rate=24000)


def pad_to_shape(max_shape, codec):
    if codec.shape[-1] == max_shape:
        return codec

    num_pad = max_shape - codec.shape[-1]
    left_pad = num_pad // 2
    right_pad = num_pad - left_pad
    
    padded_codec = F.pad(codec, (left_pad, right_pad), "constant", 0)
    padded_codec[:, :, :, :left_pad] = codec[:, :, :, :1]
    padded_codec[:, :, :, -right_pad:] = codec[:, :, :, -1:]

    return padded_codec

def encodec_test(audio_dir):
    import re
    from pathlib import Path
    from collections import defaultdict

    get_speaker = lambda name: re.findall(r"example\_(\d+)\_", name)[0]
    get_step = lambda name: int(re.findall(r"aug\_(\d+)\.wav", name)[0])

    encodec = EncodecAugModule()

    audio_dir = Path(audio_dir)
    no_aug_dict = {}
    aug_dict = defaultdict(dict)

    for p in audio_dir.glob("example_*_no_aug*"):
        speaker = get_speaker(p.name)

        inputs = encodec.load_inputs_from_file(p)
        codec = encodec.encode_to_codec(inputs)
        embed = encodec.get_embed_from_codec(codec)

        no_aug_dict[speaker] = embed
    print("complete encode no augmentation audios")
    
    for p in audio_dir.glob("example_*_step_aug*"):
        speaker = get_speaker(p.name)
        steps = get_step(p.name)

        inputs = encodec.load_inputs_from_file(p)
        codec = encodec.encode_to_codec(inputs)
        embed = encodec.get_embed_from_codec(codec)

        aug_dict[speaker][steps] = embed
    
    print("complete encode augmentated audios")

    all_speakers = list(no_aug_dict.keys())
    steps_cosine_sim_dict = defaultdict(list)

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

        aug_embeds = torch.cat(aug_embeds)

        similarities = batch_cosine_similarity(source_embed, aug_embeds)
        for index, score in zip(step_index, similarities):
            steps_cosine_sim_dict[int(index)].append(score)
    
    print("complete audios cosine similarity")
    return steps_cosine_sim_dict


if __name__ == "__main__":
    #debug()
    from pathlib import Path

    encodec = EncodecAugModule()

    test_dir = "../_examples/"
    results = encodec_test(test_dir)
    sorted_keys = sorted(list(results.keys()))
    print(sorted_keys)
    v = [np.mean(results[k]) for k in sorted_keys]
    print(v)