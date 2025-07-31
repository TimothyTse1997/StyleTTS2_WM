import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def save_melspectrogram(wav_path, output_path=None, sr=22050, n_mels=128, hop_length=512):
    """
    Plot and save the mel spectrogram of a WAV file.

    Args:
        wav_path (str): Path to the input WAV file.
        output_path (str or None): Path to save the PNG image. If None, saves in the same directory as wav_path.
        sr (int): Sampling rate to use (librosa will resample if needed).
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length for STFT.

    Returns:
        str: Path to the saved PNG file.
    """
    y, sr = librosa.load(wav_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.splitext(wav_path)[0] + "_melspec.png"

    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path

def save_line_plot(array, output_path="line_plot.png", title="Line Plot", xlabel="Index", ylabel="Value"):
    """
    Plots a line graph from a 1D array and saves it as a PNG.

    Args:
        array (list or np.ndarray): 1D array of values to plot.
        output_path (str): Path to save the PNG image.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.

    Returns:
        str: Path to the saved PNG file.
    """
    array = np.asarray(array)

    plt.figure(figsize=(8, 4))
    plt.plot(array, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path

if __name__ == "__main__":
    #arr = [np.float32(0.9715434), np.float32(0.9713861), np.float32(0.9718645), np.float32(0.97143364), np.float32(0.9698601), np.float32(0.97221655), np.float32(0.96804017), np.float32(0.9709878), np.float32(0.970409), np.float32(0.96975136), np.float32(0.9705501), np.float32(0.9698626), np.float32(0.96868855), np.float32(0.9724921), np.float32(0.97086674), np.float32(0.9700451), np.float32(0.9695116), np.float32(0.9721573), np.float32(0.97032446)]
    #save_line_plot(arr, output_path="images/encodec_cosine_similarity_per_step_plot.png", title="encodec cosine similarity per step plot", xlabel="Steps", ylabel="Cosine Similarity (Encodec)")

    #arr = [np.float32(2.4858298), np.float32(2.8969822), np.float32(2.6551948), np.float32(2.7343056), np.float32(2.767185), np.float32(2.511956), np.float32(3.0303855), np.float32(2.6071973), np.float32(2.8371124), np.float32(2.8789208), np.float32(2.9904044), np.float32(3.0855417), np.float32(2.9219482), np.float32(2.4021347), np.float32(2.8353603), np.float32(2.7815094), np.float32(2.9027252), np.float32(2.714747), np.float32(2.9250226)]
    #save_line_plot(arr, output_path="images/melspec_cosine_similarity_per_step_plot.png", title="melspec L1 per step plot", xlabel="Steps", ylabel="Mel Spec L1")

    save_melspectrogram(
        "../_examples/example_3170_no_aug.wav",
        output_path="./images/3170_no_aug_melspec.png",
    )

    save_melspectrogram(
        "../_examples/example_3170_step_aug_0.wav",
        output_path="./images/3170_step_aug_0_melspec.png",
    )
