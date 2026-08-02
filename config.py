import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration: float = 10.0
    n_samples: int = 160000


@dataclass
class FeatureConfig:
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 320
    win_length: int = 1024
    fmin: float = 500.0
    fmax: float = 3000.0
    n_mfcc: int = 40
    pcen_gain: float = 0.98
    pcen_bias: float = 2.0
    pcen_power: float = 0.5
    pcen_time_constant: float = 0.4
    pcen_eps: float = 1e-6


@dataclass
class ModelConfig:
    embedding_dim: int = 128
    num_classes: int = 2
    dropout: float = 0.3


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    patience: int = 15
    min_delta: float = 0.001
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    use_augmentation: bool = True
    mixup_alpha: float = 0.1
    spec_augment: bool = True
    use_class_weights: bool = True
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class ExperimentConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_root: str = "./data"
    output_dir: str = "./outputs"
    feature_type: str = "pcen"
    exp_name: str = "gibbon_detection"


def get_config(feature_type: str = "mel") -> ExperimentConfig:
    config = ExperimentConfig()
    config.feature_type = feature_type
    config.exp_name = f"gibbon_{feature_type}"
    return config
