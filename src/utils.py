import random
import numpy as np
import librosa
import pandas as pd
import os

def padding_audio(wav, max_signal_length):
    signal_length = wav.shape[0]
    
    if signal_length > max_signal_length:
        padding_length = signal_length - max_signal_length
        padding_length = padding_length // 2
        wav = wav[padding_length:max_signal_length+padding_length]
    else:
        padding_length = max_signal_length - signal_length
        padding_rem = padding_length % 2
        padding_length = padding_length // 2
        
        wav = np.pad(wav, (padding_length, padding_length+padding_rem), "constant", constant_values=0)
    
    return wav

def spec_augment(spec: np.ndarray, num_mask=2, 
                 freq_masking_max_percentage=0.1, time_masking_max_percentage=0.2):
    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec

def preprocess(path, max_signal_length, origin_sample_rate, target_sample_rate):
    wav, _ = librosa.load(path, sr=origin_sample_rate)
    
    wav = librosa.resample(wav, orig_sr=origin_sample_rate, 
                            target_sr=target_sample_rate)
    wav = padding_audio(wav, max_signal_length=max_signal_length)
    
    return wav

def load_data(label_path, wavs_path):
    data = pd.read_csv(
        f'{label_path}', sep="|",
        names=["id", "transcript", "label"], 
        dtype={"id":"str", "transcript":"str", "label":"str"}
        )
    data["wav_path"] = data.id.apply(lambda x:os.path.join(f'{wavs_path}', f'{x}.wav'))
    
    return data

def load_npy_data(path):
    data = np.load(path, allow_pickle=True).item()
    return data["x"].transpose(0, 2, 1), data["y"]

def extract_feature(path, data_config, general_config):
    origin_sample_rate = data_config["sample_rate"]
    target_sample_rate = general_config["sample_rate"]
    max_signal_length = general_config["max_signal_duration"] * target_sample_rate
    
    wav = preprocess(path, max_signal_length, origin_sample_rate, target_sample_rate)
    if data_config["feature"] == "mfcc":
        feature = extract_mfcc(wav, general_config["mfcc"]) 
    elif data_config["feature"] == "mel":
        feature = extract_mel_spectrogram(wav, general_config["mel"]) 
    return feature

def prepare_data(general_config, data_config):    
    if data_config["feature"] in ["mfcc", "mel"]:        
        data_df = load_data(
            label_path=data_config["label_path"],
            wavs_path=data_config["wavs_path"])
        
        if data_config["feature"] == "mfcc":
            inputs = data_df["wav_path"].apply(lambda x: extract_feature(x, data_config, general_config))
            inputs = np.stack(inputs.to_list())
            
            labels = data_df["label"].apply(lambda x: data_config["label"][x]).to_list()
            labels = np.array(labels, dtype=np.uint8)
            
            _labels = np.zeros((labels.size, labels.max() + 1))
            _labels[np.arange(labels.size), labels] = 1
            labels = _labels
            
        elif data_config["feature"] == "mel":
            data_df["feature"] = data_df["wav_path"].apply(lambda x: extract_feature(x, data_config, general_config))
            inputs = np.stack(data_df["feature"].to_list())
            
            labels = data_df["label"].apply(lambda x: data_config["label"][x]).to_list()
            labels = np.array(labels, dtype=np.uint8)
            
            _labels = np.zeros((labels.size, labels.max() + 1))
            _labels[np.arange(labels.size), labels] = 1
            labels = _labels
    else:
        inputs, labels = load_npy_data(data_config["npy_path"])
    
    return inputs, labels

def extract_mfcc(wav, config):
    mfcc = librosa.feature.mfcc(
        y=wav, 
        sr=config["sample_rate"],
        hop_length=config["hop_length"],
        win_length=config["win_length"],
        n_mfcc=config["n_mfcc"],
        fmax=config["fmax"], 
        fmin=config["fmin"])
    
    return mfcc

def extract_mel_spectrogram(wav, config):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav, 
        sr=config["sample_rate"],
        hop_length=config["hop_length"],
        win_length=config["win_length"],
        n_mels=config["n_mels"],
        fmax=config["fmax"], 
        fmin=config["fmin"],)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    
    return mel_spectrogram