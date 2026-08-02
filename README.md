# CBSkyNet

CBSkyNet is a PyTorch model for detecting gibbon calls in audio recordings. It combines a CNN10-style backbone, CBAM attention, a bidirectional LSTM, and temporal attention pooling.

## Files

```text
config.py       Experiment settings
dataset.py      Audio loading and feature extraction
model.py        CBSkyNet architecture
trainer.py      Training and evaluation
main.py         Training entry point
inference.py    Long-audio sliding-window detection
```

## Installation

```bash
pip install torch torchaudio numpy scikit-learn matplotlib tqdm
```

## Dataset structure

```text
data/
├── train/
│   ├── gibbon/
│   └── non_gibbon/
├── val/
│   ├── gibbon/
│   └── non_gibbon/
└── test/
    ├── gibbon/
    └── non_gibbon/
```

## Training

The released configuration uses 10-second audio clips, a 16 kHz sampling rate, and 64-bin PCEN features.

```bash
python main.py \
  --feature pcen \
  --data_root ./data \
  --output_dir ./outputs
```

The checkpoint is saved to:

```text
outputs/pcen_CBSkyNet/best_model.pt
```

## Inference

Set the paths at the top of `inference.py`:

```python
CHECKPOINT_PATH = "./checkpoints/best_model.pt"
AUDIO_INPUT = "./audio"
OUTPUT_DIR = "./detection_results"
FEATURE_TYPE = "pcen"
MODEL_TYPE = "CBSkyNet"
```

Then run:

```bash
python inference.py
```

## Input configuration

CBSkyNet expects 64 frequency bins. The released checkpoint should be used with the same PCEN settings used during training.

## Citation

Please cite the associated paper when using this repository. Add the final citation after publication.

## License

Add the license selected for the repository.
