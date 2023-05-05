import librosa

class Feature_Extractor():
    def __init__(self, config) -> None:
        self.config = config
    
    def extract_mfcc(self, wav):
        mfcc = librosa.feature.mfcc(
            y=wav, 
            sr=int(config["sample_rate"]),
            hop_length=int(config["hop_length"]),
            win_length=int(config["win_length"]),
            n_mfcc=int(config["n_mfcc"]),
            fmax=int(config["fmax"]), 
            fmin=int(config["fmin"]))
        
        return mfcc
    
    def extract_mel_spectrogram(self, wav):
        mel_spectrogram = librosa.feature.melspectrogram(
            y=wav, 
            sr=int(self.config["mel"]["sample_rate"]),
            hop_length=int(self.config["mel"]["hop_length"]),
            win_length=int(self.config["mel"]["win_length"]),
            n_mels=int(self.config["mel"]["n_mels"]),
            fmax=int(self.config["mel"]["fmax"]), 
            fmin=int(self.config["mel"]["fmin"]),)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        return mel_spectrogram

if __name__ == "__main__":
    from yaml.loader import SafeLoader
    import yaml

    with open("configs/general_config.yml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)
    feature_extractor = Feature_Extractor(config)