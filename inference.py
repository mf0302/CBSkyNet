import os
import re
import sys
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import json

from config import get_config
from dataset import AudioFeatureExtractor, normalize_features
from model import get_model, ALL_MODEL_TYPES

CHECKPOINT_PATH = "./checkpoints/best_model.pt"

AUDIO_INPUT = "./audio"

OUTPUT_DIR = "./detection_results"

FEATURE_TYPE = "pcen"
MODEL_TYPE = "CBSkyNet"
DEVICE = "cuda"

WINDOW_SEC = 10
HOP_SEC = 5.0
THRESHOLD = 0.5
MERGE_GAP_SEC = 2.0
INFER_BATCH_SIZE = 16

EXPORT_CLIPS = True
PADDING_SEC = 1.0

SAVE_JSON = True


def parse_filename(filename: str) -> Optional[Dict]:
    stem = Path(filename).stem

    pattern = r'^(.+)_(\d{8})_(\d{6})$'
    m = re.match(pattern, stem)
    if not m:
        print(f"[WARN] 无法解析文件名: {filename}，将不附加绝对时间戳")
        return None

    device_id, date_str, time_str = m.group(1), m.group(2), m.group(3)

    try:
        start_datetime = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    except ValueError as e:
        print(f"[WARN] 时间解析失败 ({date_str}_{time_str}): {e}")
        return None

    return {
        'device_id':      device_id,
        'date_str':       date_str,
        'time_str':       time_str,
        'start_datetime': start_datetime,
    }


def offset_to_abs_time(start_dt: datetime, offset_sec: float) -> str:
    abs_dt = start_dt + timedelta(seconds=offset_sec)
    return abs_dt.strftime("%H%M%S")


def build_output_name(meta: Dict, start_sec: float, end_sec: float) -> str:
    start_tag = offset_to_abs_time(meta['start_datetime'], start_sec)
    end_tag   = offset_to_abs_time(meta['start_datetime'], end_sec)
    return f"{meta['device_id']}_{meta['date_str']}_{start_tag}_{end_tag}"


class GibbonDetector:
    def __init__(
        self,
        checkpoint_path: str,
        feature_type: str = 'pcen',
        model_type: str = 'CBSkyNet',
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.config       = get_config(feature_type)
        self.feature_type = feature_type
        self.model_type   = model_type

        self.feature_extractor = AudioFeatureExtractor(self.config)

        self.model = get_model(self.config, model_type)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"[GibbonDetector] checkpoint : {checkpoint_path}")
        print(f"[GibbonDetector] feature    : {feature_type}")
        print(f"[GibbonDetector] model      : {model_type}")
        print(f"[GibbonDetector] device     : {self.device}")

    def _load_and_pad(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.config.audio.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.audio.sample_rate)
            waveform  = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_len = self.config.audio.n_samples
        if waveform.shape[-1] > target_len:
            waveform = waveform[..., :target_len]
        elif waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))

        return waveform

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(waveform)
        feat = normalize_features(feat, self.feature_type)
        return feat

    @torch.no_grad()
    def _infer_batch(self, feature_batch: torch.Tensor) -> torch.Tensor:
        feature_batch = feature_batch.to(self.device)
        outputs = self.model(feature_batch)
        probs   = F.softmax(outputs['logits'].float(), dim=1)
        return probs[:, 1].cpu()

    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict:
        waveform, sr = torchaudio.load(audio_path)
        waveform     = self._load_and_pad(waveform, sr)
        feat         = self._extract_features(waveform).unsqueeze(0)
        prob         = self._infer_batch(feat).item()
        pred_class   = int(prob >= 0.5)

        return {
            'path':                  audio_path,
            'prediction':            'gibbon' if pred_class else 'non_gibbon',
            'gibbon_probability':    prob,
            'non_gibbon_probability': 1 - prob,
            'confidence':            prob if pred_class else 1 - prob,
        }

    @torch.no_grad()
    def predict_batch(self, audio_paths: List[str], batch_size: int = 32) -> List[Dict]:
        results = []
        for i in tqdm(range(0, len(audio_paths), batch_size), desc="Predicting"):
            batch_paths = audio_paths[i: i + batch_size]
            feats = []
            for path in batch_paths:
                waveform, sr = torchaudio.load(path)
                waveform     = self._load_and_pad(waveform, sr)
                feats.append(self._extract_features(waveform))

            feat_batch = torch.stack(feats)
            probs      = self._infer_batch(feat_batch)

            for path, prob in zip(batch_paths, probs.tolist()):
                pred_class = int(prob >= 0.5)
                results.append({
                    'path':                   path,
                    'prediction':             'gibbon' if pred_class else 'non_gibbon',
                    'gibbon_probability':     prob,
                    'non_gibbon_probability': 1 - prob,
                    'confidence':             prob if pred_class else 1 - prob,
                })
        return results

    @torch.no_grad()
    def detect_long_audio(
        self,
        audio_path: str,
        window_sec: float     = 10.0,
        hop_sec: float        = 5.0,
        threshold: float      = 0.5,
        merge_gap_sec: float  = 2.0,
        infer_batch_size: int = 16,
    ) -> Dict:
        print(f"\n[detect] 加载音频: {audio_path}")
        waveform, sr = torchaudio.load(audio_path)

        target_sr = self.config.audio.sample_rate
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform  = resampler(waveform)
            sr        = target_sr

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform     = waveform.squeeze(0)
        total_samples = waveform.shape[0]
        duration_sec  = total_samples / sr
        print(f"[detect] 音频时长: {duration_sec/60:.1f} 分钟 ({duration_sec:.1f} 秒)")

        meta = parse_filename(audio_path)

        window_samples = int(window_sec * sr)
        hop_samples    = int(hop_sec    * sr)

        starts = list(range(0, total_samples - window_samples + 1, hop_samples))
        if not starts:
            print("[detect] 音频过短，无法进行滑动窗口检测")
            return {
                'path': audio_path, 'duration': duration_sec,
                'meta': meta, 'raw_detections': [], 'detections': []
            }

        print(f"[detect] 滑动窗口: {window_sec}s 窗口 / {hop_sec}s 步进 → {len(starts)} 个窗口")

        raw_detections = []

        for batch_start in tqdm(
            range(0, len(starts), infer_batch_size),
            desc="滑动窗口推理",
            unit="batch"
        ):
            batch_starts = starts[batch_start: batch_start + infer_batch_size]
            feats = []

            for s in batch_starts:
                clip = waveform[s: s + window_samples].unsqueeze(0)

                if clip.shape[-1] < window_samples:
                    clip = F.pad(clip, (0, window_samples - clip.shape[-1]))

                feat = self._extract_features(clip)
                feats.append(feat)

            feat_batch = torch.stack(feats)
            probs      = self._infer_batch(feat_batch)

            for s, prob in zip(batch_starts, probs.tolist()):
                start_sec = s / sr
                end_sec   = (s + window_samples) / sr
                if prob >= threshold:
                    raw_detections.append({
                        'start_sec': start_sec,
                        'end_sec':   end_sec,
                        'prob':      prob,
                    })

        print(f"[detect] 阈值过滤后原始检测数: {len(raw_detections)}")

        merged = self._merge_detections(raw_detections, gap_sec=merge_gap_sec)
        print(f"[detect] 合并后检测数: {len(merged)}")

        for det in merged:
            if meta:
                det['start_tag']   = offset_to_abs_time(meta['start_datetime'], det['start_sec'])
                det['end_tag']     = offset_to_abs_time(meta['start_datetime'], det['end_sec'])
                det['output_name'] = build_output_name(meta, det['start_sec'], det['end_sec'])
            else:
                det['start_tag']   = f"{int(det['start_sec']):06d}s"
                det['end_tag']     = f"{int(det['end_sec']):06d}s"
                det['output_name'] = (
                    f"{Path(audio_path).stem}"
                    f"_{int(det['start_sec'])}_{int(det['end_sec'])}"
                )

        return {
            'path':           audio_path,
            'duration':       duration_sec,
            'meta':           meta,
            'raw_detections': raw_detections,
            'detections':     merged,
        }

    @staticmethod
    def _merge_detections(
        detections: List[Dict],
        gap_sec: float = 2.0
    ) -> List[Dict]:
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda x: x['start_sec'])
        merged = [{
            'start_sec': sorted_dets[0]['start_sec'],
            'end_sec':   sorted_dets[0]['end_sec'],
            'max_prob':  sorted_dets[0]['prob'],
            'n_windows': 1,
        }]

        for det in sorted_dets[1:]:
            last = merged[-1]
            if det['start_sec'] - last['end_sec'] <= gap_sec:
                last['end_sec']  = max(last['end_sec'], det['end_sec'])
                last['max_prob'] = max(last['max_prob'], det['prob'])
                last['n_windows'] += 1
            else:
                merged.append({
                    'start_sec': det['start_sec'],
                    'end_sec':   det['end_sec'],
                    'max_prob':  det['prob'],
                    'n_windows': 1,
                })

        return merged

    @staticmethod
    def export_clips(
        audio_path: str,
        detections: List[Dict],
        output_dir: str,
        padding_sec: float = 1.0,
    ) -> List[str]:
        if not detections:
            print("[export] 没有检测到任何片段，跳过导出")
            return []

        os.makedirs(output_dir, exist_ok=True)

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        total_samples = waveform.shape[-1]
        saved_paths   = []

        for det in detections:
            start_sample = max(0, int((det['start_sec'] - padding_sec) * sr))
            end_sample   = min(total_samples, int((det['end_sec'] + padding_sec) * sr))
            clip         = waveform[:, start_sample: end_sample]

            out_name = det.get('output_name', f"clip_{int(det['start_sec'])}_{int(det['end_sec'])}")
            out_path = os.path.join(output_dir, out_name + ".wav")

            torchaudio.save(out_path, clip, sr)
            saved_paths.append(out_path)
            print(f"  [export] 已保存: {out_path}  ({clip.shape[-1]/sr:.1f}s)")

        return saved_paths


def main():
    if FEATURE_TYPE not in ['mel', 'mfcc', 'fbank', 'pcen']:
        raise ValueError(f"FEATURE_TYPE 设置错误: {FEATURE_TYPE}，可选: mel / mfcc / fbank / pcen")

    if MODEL_TYPE not in ALL_MODEL_TYPES:
        raise ValueError(f"MODEL_TYPE 设置错误: {MODEL_TYPE}，当前支持: {ALL_MODEL_TYPES}")

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型权重不存在，请检查 CHECKPOINT_PATH: {checkpoint_path}")

    audio_path = Path(AUDIO_INPUT)
    if not audio_path.exists():
        raise FileNotFoundError(f"输入音频或文件夹不存在，请检查 AUDIO_INPUT: {audio_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("天行长臂猿鸣声检测推理")
    print("=" * 60)
    print(f"模型权重     : {CHECKPOINT_PATH}")
    print(f"输入路径     : {AUDIO_INPUT}")
    print(f"输出目录     : {OUTPUT_DIR}")
    print(f"特征类型     : {FEATURE_TYPE}")
    print(f"模型类型     : {MODEL_TYPE}")
    print(f"检测窗口     : {WINDOW_SEC}s")
    print(f"窗口步进     : {HOP_SEC}s")
    print(f"检测阈值     : {THRESHOLD}")
    print(f"合并间隔     : {MERGE_GAP_SEC}s")
    print(f"导出片段     : {EXPORT_CLIPS}")
    print(f"保存 JSON    : {SAVE_JSON}")
    print("=" * 60)

    detector = GibbonDetector(
        checkpoint_path=str(checkpoint_path),
        feature_type=FEATURE_TYPE,
        model_type=MODEL_TYPE,
        device=DEVICE,
    )

    if audio_path.is_dir():
        audio_files = sorted(audio_path.glob('*.wav'))
        print(f"\n[main] 目录模式，共找到 {len(audio_files)} 个 wav 文件")
    else:
        if audio_path.suffix.lower() != '.wav':
            raise ValueError(f"输入文件不是 .wav 格式: {audio_path}")
        audio_files = [audio_path]
        print("\n[main] 单文件模式")

    if not audio_files:
        print(f"[main] 没有找到 wav 文件: {audio_path}")
        return

    all_results = []

    for wav_file in audio_files:
        result = detector.detect_long_audio(
            audio_path=str(wav_file),
            window_sec=WINDOW_SEC,
            hop_sec=HOP_SEC,
            threshold=THRESHOLD,
            merge_gap_sec=MERGE_GAP_SEC,
            infer_batch_size=INFER_BATCH_SIZE,
        )
        all_results.append(result)

        print(f"\n{'─'*60}")
        print(f"文件   : {wav_file.name}")
        print(f"时长   : {result['duration']/60:.1f} min")
        print(f"检测数 : {len(result['detections'])}")
        if result['detections']:
            print(f"{'开始时间':>10}  {'结束时间':>10}  {'最大概率':>8}  {'窗口数':>6}  输出文件名")
            print(f"{'─'*60}")
            for det in result['detections']:
                print(
                    f"  {det['start_tag']:>10}  {det['end_tag']:>10}"
                    f"  {det['max_prob']:>8.4f}  {det['n_windows']:>6}"
                    f"  {det['output_name']}.wav"
                )

        if EXPORT_CLIPS and result['detections']:
            clip_dir = os.path.join(OUTPUT_DIR, 'clips')
            detector.export_clips(
                audio_path=str(wav_file),
                detections=result['detections'],
                output_dir=clip_dir,
                padding_sec=PADDING_SEC,
            )

        if SAVE_JSON:
            json_name = wav_file.stem + "_detections.json"
            json_path = os.path.join(OUTPUT_DIR, json_name)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'file':       str(wav_file),
                    'duration':   result['duration'],
                    'detections': result['detections'],
                }, f, indent=2, ensure_ascii=False)
            print(f"[main] JSON 结果已保存: {json_path}")

    total_detections = sum(len(r['detections']) for r in all_results)
    print(f"\n{'='*60}")
    print(f"处理文件数 : {len(all_results)}")
    print(f"检测片段总数: {total_detections}")
    print(f"输出目录   : {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
