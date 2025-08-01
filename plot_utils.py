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
    #arr = [np.float32(0.99933124), np.float32(0.9993057), np.float32(0.9992981), np.float32(0.9993009), np.float32(0.9993216), np.float32(0.99925894), np.float32(0.9992607), np.float32(0.9993043), np.float32(0.999252), np.float32(0.9993219), np.float32(0.9992284), np.float32(0.999294), np.float32(0.9992735), np.float32(0.9993488), np.float32(0.9993084), np.float32(0.9993332), np.float32(0.9993253), np.float32(0.9992544), np.float32(0.99928164)]
    #save_line_plot(arr, output_path="images/encodec_cosine_similarity_per_step_plot.png", title="encodec cosine similarity per step plot", xlabel="Steps", ylabel="Cosine Similarity (Encodec)")

    #arr = [np.float32(0.07977717), np.float32(0.07920804), np.float32(0.08648226), np.float32(0.10145918), np.float32(0.070338435), np.float32(0.07417515), np.float32(0.08902883), np.float32(0.08351044), np.float32(0.085272744), np.float32(0.07908802), np.float32(0.08136624), np.float32(0.08352053), np.float32(0.08429008), np.float32(0.078603536), np.float32(0.08145918), np.float32(0.075814374), np.float32(0.08098524), np.float32(0.08521092), np.float32(0.08416409)]
    #save_line_plot(arr, output_path="images/melspec_cosine_similarity_per_step_plot.png", title="melspec L1 per step plot", xlabel="Steps", ylabel="Mel Spec L1")

    save_melspectrogram(
        "../_examples/example_3170_no_aug.wav",
        output_path="./images/3170_no_aug_melspec.png",
    )

    save_melspectrogram(
        "../_examples/example_3170_step_aug_0.wav",
        output_path="./images/3170_step_aug_0_melspec.png",
    )
