"""
This module contains functions for loading and preprocessing
data before it is fed into the genre classifier.

Additional notes:

The format of the loaded audio file does not significantly impact
the extraction of the features from the loaded data because the
library used for loading (librosa) will convert the audio into
a NumPy Array. It is important to note however, that it is best
to use lossless formats (WAV, FLAC) for the highest quality of
features extraction. Using lossy formats such as MP3 or OGG can
have an impact on the feature extraction and the quality of prediction.

The sampling rate for the training data was 22050 Hz.
"""
import librosa
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict


SAMPLING_RATE = 22050


def load_audio(file_path: str) -> Tuple[NDArray[np.float32], int]:
    """
    Load an audio file and return the audio data and
    sampling rate.
    
    Parameters:
        file_path: The path to the audio file.
        
    Returns:
        A tuple containing audio data as a numpy array and the
            sampling rate.
    
    Raises:
        IOError: If the file cannot be loaded.
    """
    try:
        audio_data, sr = librosa.load(file_path, sr=SAMPLING_RATE)
        return audio_data, sr
    except Exception as e:
        raise IOError(f"Error loading {file_path}: {e}")



def get_sampling_rate(file_path: str) -> int:
    """
    Extract and return the sampling rate of the audio
    file.
    
    Parameters:
        file_path: The path to the audio file.
        
    Returns:
        The sampling rate of the audio file.
        
    Raises:
        IOError: If the file is not found.
    """
    try:
        _, sr = librosa.load(file_path, sr=None)
        return sr
    except Exception as e:
        raise IOError(f"Error loading {file_path}: {e}")


def extract_features(audio_data: NDArray[np.float32]) -> Dict[str, float]:
    """
    Extract features from audio data.
    
    Parameters:
        audio_data: The audio data as a NumPy array.
        
    Returns:
        A dictionary of extracted features.
    """
    features = {}
    
    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=SAMPLING_RATE)
    features["chroma_stft_mean"] = np.mean(chroma_stft)
    features["chroma_stft_var"] = np.var(chroma_stft)
    
    # RMS
    rms = librosa.feature.rms(y=audio_data)
    features["rms_mean"] = np.mean(rms)
    features["rms_var"] = np.var(rms)
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=SAMPLING_RATE)
    features["spectral_centroid_mean"] = np.mean(spectral_centroid)
    features["spectral_centroid_var"] = np.var(spectral_centroid)
    
    # Spectral Bandwidth
    spectral_bandwith = librosa.feature.spectral_bandwidth(y=audio_data, sr=SAMPLING_RATE)
    features["spectral_bandwidth_mean"] = np.mean(spectral_bandwith)
    features["spectral_bandwidth_var"] = np.var(spectral_bandwith)
    
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=SAMPLING_RATE)
    features["rolloff_mean"] = np.mean(rolloff)
    features["rolloff_var"] = np.var(rolloff)
    
    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)
    features["zero_crossing_rate_mean"] = np.mean(zero_crossing_rate)
    features["zero_crossing_rate_var"] = np.var(zero_crossing_rate)
    
    # Harmony and Perceived Harmonicity
    harmonic, perceptr = librosa.effects.hpss(audio_data)
    features["harmony_mean"] = np.mean(harmonic)
    features["harmony_var"] = np.var(harmonic)
    features["perceptr_mean"] = np.mean(perceptr)
    features["perceptr_var"] = np.var(perceptr)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=SAMPLING_RATE)
    features["tempo"] = tempo
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLING_RATE, n_mfcc=20)
    for i in range(20):
        features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc{i+1}_var"] = np.var(mfccs[i])
        
    return features
