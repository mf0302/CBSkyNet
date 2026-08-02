import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
from pathlib import Path

from config import ExperimentConfig


class AudioFeatureExtractor:
    def __init__(self, config: ExperimentConfig):
        self.audio_cfg = config.audio
        self.feat_cfg = config.feature
        self.feature_type = config.feature_type

        self._init_extractors()

    def _init_extractors(self):
        self.mel_extractor = T.MelSpectrogram(
            sample_rate=self.audio_cfg.sample_rate,
            n_fft=self.feat_cfg.n_fft,
            hop_length=self.feat_cfg.hop_length,
            win_length=self.feat_cfg.win_length,
            n_mels=self.feat_cfg.n_mels,
            f_min=self.feat_cfg.fmin,
            f_max=self.feat_cfg.fmax,
            power=2.0
        )

        self.mfcc_extractor = T.MFCC(
            sample_rate=self.audio_cfg.sample_rate,
            n_mfcc=self.feat_cfg.n_mfcc,
            melkwargs={
                "n_fft": self.feat_cfg.n_fft,
                "hop_length": self.feat_cfg.hop_length,
                "win_length": self.feat_cfg.win_length,
                "n_mels": self.feat_cfg.n_mels,
                "f_min": self.feat_cfg.fmin,
                "f_max": self.feat_cfg.fmax,
            }
        )

        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def extract_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_extractor(waveform)

        mel_spec = torch.clamp(mel_spec, min=1e-10)

        log_mel = self.amplitude_to_db(mel_spec)

        log_mel = torch.nan_to_num(
            log_mel,
            nan=-80.0,
            posinf=0.0,
            neginf=-80.0
        )

        return log_mel

    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        mfcc = self.mfcc_extractor(waveform)

        mfcc_delta = torchaudio.functional.compute_deltas(mfcc)
        mfcc_delta2 = torchaudio.functional.compute_deltas(mfcc_delta)

        mfcc_features = torch.cat([mfcc, mfcc_delta, mfcc_delta2], dim=1)

        return mfcc_features

    def extract_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_extractor(waveform)
        fbank = torch.sqrt(mel_spec + 1e-10)

        return fbank

    def extract_pcen(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_extractor(waveform)

        sr = self.audio_cfg.sample_rate
        hop = self.feat_cfg.hop_length

        gain = self.feat_cfg.pcen_gain
        bias = self.feat_cfg.pcen_bias
        power = self.feat_cfg.pcen_power
        time_constant = self.feat_cfg.pcen_time_constant
        eps = self.feat_cfg.pcen_eps

        s = 1.0 - torch.exp(
            torch.tensor(-1.0 / (time_constant * sr / hop), device=mel_spec.device)
        )

        E = mel_spec
        n_time = E.shape[-1]

        M = torch.zeros_like(E)
        M[..., 0] = E[..., 0]

        for t in range(1, n_time):
            M[..., t] = (1.0 - s) * M[..., t - 1] + s * E[..., t]

        smooth = torch.pow(M + eps, gain)

        pcen = torch.pow(E / smooth + bias, power) - torch.pow(
            torch.tensor(bias, device=mel_spec.device),
            power
        )

        pcen = torch.nan_to_num(pcen, nan=0.0, posinf=0.0, neginf=0.0)

        return pcen

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.feature_type == "mel":
            return self.extract_mel(waveform)
        elif self.feature_type == "mfcc":
            return self.extract_mfcc(waveform)
        elif self.feature_type == "fbank":
            return self.extract_fbank(waveform)
        elif self.feature_type == "pcen":
            return self.extract_pcen(waveform)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")


class SpecAugment:
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_freq_masks):
            spec = self.freq_mask(spec)

        for _ in range(self.num_time_masks):
            spec = self.time_mask(spec)

        return spec


class AudioAugment:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def add_noise(self, waveform: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
        noise = torch.randn_like(waveform)

        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()

        snr = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr * noise_power + 1e-10))

        return waveform + scale * noise

    def time_shift(self, waveform: torch.Tensor, shift_max: float = 0.2) -> torch.Tensor:
        shift = int(random.uniform(-shift_max, shift_max) * waveform.shape[-1])
        return torch.roll(waveform, shifts=shift, dims=-1)

    def speed_perturb(
        self,
        waveform: torch.Tensor,
        rate_range: Tuple[float, float] = (0.9, 1.1)
    ) -> torch.Tensor:
        rate = random.uniform(*rate_range)
        effects = [["speed", str(rate)], ["rate", str(self.sample_rate)]]

        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            self.sample_rate,
            effects,
            channels_first=True
        )

        target_len = waveform.shape[-1]

        if augmented.shape[-1] > target_len:
            augmented = augmented[..., :target_len]
        elif augmented.shape[-1] < target_len:
            augmented = torch.nn.functional.pad(
                augmented,
                (0, target_len - augmented.shape[-1])
            )

        return augmented

    def gain(self, waveform: torch.Tensor, gain_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        gain = random.uniform(*gain_range)
        return waveform * gain

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            waveform = self.add_noise(waveform, snr_db=random.uniform(15, 30))

        if random.random() < 0.3:
            waveform = self.time_shift(waveform)

        if random.random() < 0.3:
            waveform = self.gain(waveform)

        return waveform


def normalize_features(features: torch.Tensor, feature_type: str) -> torch.Tensor:
    if feature_type == "pcen":
        fmin = features.min()
        fmax = features.max()

        if fmax - fmin > 1e-10:
            features = (features - fmin) / (fmax - fmin)

        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    else:
        mean = features.mean()
        std = features.std()

        features = (features - mean) / (std + 1e-10)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features


class GibbonDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        config: ExperimentConfig,
        augment: bool = False
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.config = config

        self.augment = augment and split == "train"

        self.feature_extractor = AudioFeatureExtractor(config)

        if self.augment:
            self.audio_augment = AudioAugment(config.audio.sample_rate)
            self.spec_augment = SpecAugment()

        self.samples = self._load_samples()
        self.class_weights = self._compute_class_weights()

        print(
            f"[{split}] Loaded {len(self.samples)} samples, "
            f"Gibbon: {sum(1 for s in self.samples if s[1] == 1)}, "
            f"Non-Gibbon: {sum(1 for s in self.samples if s[1] == 0)}"
        )

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        split_dir = self.data_root / self.split

        gibbon_dir = split_dir / "gibbon"
        if gibbon_dir.exists():
            for f in gibbon_dir.glob("*.wav"):
                samples.append((str(f), 1))

        non_gibbon_dir = split_dir / "non_gibbon"
        if non_gibbon_dir.exists():
            for f in non_gibbon_dir.glob("*.wav"):
                samples.append((str(f), 0))

        random.shuffle(samples)

        return samples

    def _compute_class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]

        if len(labels) == 0:
            return torch.tensor([1.0, 1.0], dtype=torch.float32)

        class_counts = np.bincount(labels, minlength=2)
        total = len(labels)

        weights = total / (len(class_counts) * class_counts + 1e-10)

        return torch.tensor(weights, dtype=torch.float32)

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)

        if sr != self.config.audio.sample_rate:
            resampler = T.Resample(sr, self.config.audio.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_len = self.config.audio.n_samples

        if waveform.shape[-1] > target_len:
            if self.split == "train":
                start = random.randint(0, waveform.shape[-1] - target_len)
            else:
                start = (waveform.shape[-1] - target_len) // 2

            waveform = waveform[..., start:start + target_len]

        elif waveform.shape[-1] < target_len:
            pad_len = target_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        waveform = self._load_audio(path)

        if self.augment:
            waveform = self.audio_augment(waveform)

        features = self.feature_extractor(waveform)

        features = normalize_features(features, self.config.feature_type)

        if self.augment and self.config.train.spec_augment:
            features = self.spec_augment(features)

        return features, label


def get_dataloaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    train_dataset = GibbonDataset(
        config.data_root,
        "train",
        config,
        augment=config.train.use_augmentation
    )

    val_dataset = GibbonDataset(
        config.data_root,
        "val",
        config,
        augment=False
    )

    test_dataset = GibbonDataset(
        config.data_root,
        "test",
        config,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.class_weights

if __name__ == "__main__":
    from config import get_config

    config = get_config("mel")
    config.data_root = "./data"

    dummy_waveform = torch.randn(1, 160000)

    for feat_type in ["mel", "mfcc", "fbank", "pcen"]:
        config.feature_type = feat_type
        extractor = AudioFeatureExtractor(config)

        feat = extractor(dummy_waveform)
        normed = normalize_features(feat, feat_type)

        print(
            f"{feat_type}: raw {feat.shape} "
            f"range [{feat.min():.3f}, {feat.max():.3f}] "
            f"→ normed range [{normed.min():.3f}, {normed.max():.3f}]"
        )
