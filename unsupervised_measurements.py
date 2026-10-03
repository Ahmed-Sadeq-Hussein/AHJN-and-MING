"""unsupervised_measurements.py | Automatic metrics and the reward built from them.

Everything in this file is computed by the machine alone: no human ratings, no
character labels. (Human ratings and character labels live in
supervised_measurements.py.)

The one function most code needs
--------------------------------
    from unsupervised_measurements import voice_reward

    score = voice_reward("clip.wav", "neutral.wav", text="I would never sell you fake gold.")
    # -> an int from 0 to 100

    r = voice_reward("clip.wav", "neutral.wav", text="...", details=True)
    # -> a Reward with r.score (int), r.value (float 0..1), r.parts, r.metrics, r.skipped

What goes into the reward
-------------------------
    reward = validity * intelligibility * same_speaker * expressiveness

    validity         six cheap checks: duration, silence, voicing, emphasis sense,
                     pause sense, parsimony                         (signal processing)
    intelligibility  word error rate of a speech recogniser vs the text   (needs Whisper)
    same_speaker     speaker similarity to the reference clip         (needs Resemblyzer)
    expressiveness   how far the delivery is from what was heard before, through a
                     Wundt curve: moderate novelty scores best     (signal processing)

The first three are GUARDS: they are 1.0 for any acceptable clip and only drop when
something is wrong. Only expressiveness pulls the search anywhere. Rewarding low WER
or high similarity directly would pull every clip back to the neutral delivery.

Command line
------------
    python unsupervised_measurements.py reward clip.wav --neutral neutral.wav --text "the line"
    python unsupervised_measurements.py measure clips/ --neutral clips/neutral.wav --out calib.csv

Dependencies
------------
    always:    numpy, soundfile, praat-parselmouth
    optional:  faster-whisper (or openai-whisper)  -> intelligibility guard
               resemblyzer                          -> same-speaker guard
    A missing optional library never crashes a run: that guard is skipped, counted as
    passed, and named in Reward.skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

# --------------------------------------------------------------------------- #
# 1. Settings
# --------------------------------------------------------------------------- #

FEATURE_NAMES = ["tempo", "pause", "pitch_level", "pitch_range", "loud_dyn", "brightness"]

# One "typical step of change" per feature, used until you calibrate your own with
# fit_scale(). Rough values: 15% tempo, 10 points of pause share, 2 semitones, and so on.
DEFAULT_SCALE = np.array([0.15, 0.10, 2.0, 0.30, 0.25, 0.15])


@dataclass
class Config:
    """Every threshold and weight in one place. Change values here, not in the code."""

    # framing
    frame_s: float = 0.025
    hop_s: float = 0.010
    speech_db_below_p95: float = 30.0   # a frame is speech when this close to the loud frames
    abs_floor_db: float = -60.0
    min_pause_s: float = 0.10           # shorter silences are stop consonants, not pauses
    pitch_floor: float = 65.0
    pitch_ceiling: float = 500.0
    max_formant_hz: float = 5500.0      # about 5000 for deep voices

    # validity checks: (full score at, zero at)
    dur_full: tuple[float, float] = (0.6, 2.0)
    dur_zero: tuple[float, float] = (0.4, 3.0)
    silence_full: float = 0.40
    silence_zero: float = 0.60
    min_voiced_share: float = 0.30
    parsimony_free_ops: int = 5
    parsimony_max_ops: int = 8
    parsimony_floor: float = 0.7

    # intelligibility guard (word error rate, lower is better)
    wer_full: float = 0.10              # up to 10% wrong words costs nothing
    wer_zero: float = 0.50              # half the words wrong scores zero
    whisper_size: str = "base"
    language: str = "en"

    # same-speaker guard (cosine similarity, higher is better). CALIBRATE THESE:
    # measure a few clips you judge "same speaker" and "different speaker" first.
    sim_full: float = 0.75
    sim_zero: float = 0.55

    # expressiveness
    knn: int = 5
    wundt_n0: float = 1.5               # preferred novelty. Set from calibration
    wundt_sigma: float = 0.75           # rule of thumb: n0 / 2


CFG = Config()

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from", "as",
    "and", "or", "but", "nor", "so", "if", "than", "that", "this", "these", "those",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "it", "its", "i", "you", "he", "she", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "our", "their",
}
PUNCT = ",.;:!?"

# --------------------------------------------------------------------------- #
# 2. Acoustic features: absolute numbers measured from one clip
# --------------------------------------------------------------------------- #


@dataclass
class RawFeatures:
    # delivery (these feed the behaviour vector)
    duration_s: float        # clip length after trimming silence at both ends
    speech_s: float          # duration minus pauses
    pause_share: float       # pauses / duration, 0 to 1
    f0_median_hz: float      # how deep or high the voice is
    f0_range_st: float       # 5th to 95th percentile of pitch, in semitones
    loud_dyn_db: float       # std of frame loudness over speech frames
    centroid_hz: float       # spectral centroid: bright/clear vs dark/husky
    voiced_share: float      # voiced time / speech time. Low = pitch numbers unreliable
    # voice quality (for the voice map; mostly fixed by the reference voice)
    hnr_db: float = math.nan            # harmonics-to-noise: low = breathy, husky
    jitter: float = math.nan            # pitch wobble between cycles: high = rough
    shimmer: float = math.nan           # loudness wobble between cycles: high = rough
    f1_hz: float = math.nan
    f2_hz: float = math.nan
    f3_hz: float = math.nan
    f4_hz: float = math.nan
    formant_dispersion_hz: float = math.nan  # mean formant spacing: low = large speaker

    def to_dict(self) -> dict:
        return asdict(self)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a wav as mono float64 in [-1, 1]."""
    y, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return y.mean(axis=1), int(sr)


def _frames(y: np.ndarray, sr: int, cfg: Config) -> tuple[np.ndarray, int, int]:
    n, hop = int(round(cfg.frame_s * sr)), int(round(cfg.hop_s * sr))
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return np.lib.stride_tricks.sliding_window_view(y, n)[::hop], n, hop


def _speech_mask(db: np.ndarray, cfg: Config) -> np.ndarray:
    thr = max(np.percentile(db, 95) - cfg.speech_db_below_p95, cfg.abs_floor_db)
    return db > thr


def _pause_mask(speech: np.ndarray, cfg: Config) -> np.ndarray:
    """True for frames inside a silent run that is long enough to be a pause."""
    min_frames = int(round(cfg.min_pause_s / cfg.hop_s))
    pause = np.zeros_like(speech, dtype=bool)
    i, n = 0, len(speech)
    while i < n:
        if not speech[i]:
            j = i
            while j < n and not speech[j]:
                j += 1
            if j - i >= min_frames:
                pause[i:j] = True
            i = j
        else:
            i += 1
    return pause


def extract_raw(audio: str | Path | tuple[np.ndarray, int], cfg: Config = CFG) -> RawFeatures:
    """Measure one clip. `audio` is a wav path or a (samples, sample_rate) pair."""
    y, sr = load_audio(audio) if isinstance(audio, (str, Path)) else audio
    y = np.asarray(y, dtype="float64")

    # loudness per frame, then trim silence at both ends
    fr, n, hop = _frames(y, sr, cfg)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-10)
    speech = _speech_mask(db, cfg)
    if not speech.any():
        raise ValueError("clip contains no speech above the silence threshold")
    first, last = np.flatnonzero(speech)[[0, -1]]
    fr, db, speech = fr[first:last + 1], db[first:last + 1], speech[first:last + 1]
    y_trim = y[first * hop: last * hop + n]

    # pauses and timing
    pause = _pause_mask(speech, cfg)
    duration_s = len(speech) * cfg.hop_s
    speech_s = (~pause).sum() * cfg.hop_s

    # loudness dynamics and brightness, over the speech frames only
    loud_dyn_db = float(db[speech].std())
    mag = np.abs(np.fft.rfft(fr[speech] * np.hanning(n), axis=1))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    centroid = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + 1e-10)
    centroid_hz = float(np.average(centroid, weights=mag.sum(axis=1)))

    # pitch (Praat, through parselmouth)
    snd = parselmouth.Sound(y_trim, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=cfg.hop_s, pitch_floor=cfg.pitch_floor,
                         pitch_ceiling=cfg.pitch_ceiling)
    f0_all = pitch.selected_array["frequency"]
    voiced_times = pitch.xs()[f0_all > 0]
    f0 = f0_all[f0_all > 0]
    if len(f0) >= 5:
        f0_median = float(np.median(f0))
        lo, hi = np.percentile(f0, [5, 95])
        f0_range_st = float(12 * np.log2(hi / lo))
    else:
        f0_median, f0_range_st = math.nan, math.nan
    voiced_share = float(min(1.0, len(f0) * cfg.hop_s / max(speech_s, 1e-6)))

    raw = RawFeatures(duration_s=float(duration_s), speech_s=float(speech_s),
                      pause_share=float(pause.mean()), f0_median_hz=f0_median,
                      f0_range_st=f0_range_st, loud_dyn_db=loud_dyn_db,
                      centroid_hz=centroid_hz, voiced_share=voiced_share)
    _add_voice_quality(raw, snd, voiced_times, cfg)
    return raw


def _add_voice_quality(raw: RawFeatures, snd: parselmouth.Sound, voiced_times: np.ndarray,
                       cfg: Config) -> None:
    """HNR, jitter, shimmer, formants. Failures leave NaN and never stop a run."""
    try:
        h = snd.to_harmonicity_cc(time_step=cfg.hop_s, minimum_pitch=cfg.pitch_floor).values
        h = h[h > -100]  # Praat marks unvoiced frames with -200
        raw.hnr_db = float(h.mean()) if h.size else math.nan
    except Exception:
        pass
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", cfg.pitch_floor, cfg.pitch_ceiling)
        raw.jitter = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        raw.shimmer = float(call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        pass
    try:
        fm = snd.to_formant_burg(time_step=cfg.hop_s, max_number_of_formants=5,
                                 maximum_formant=cfg.max_formant_hz)
        # formants are only meaningful while the voice is sounding, so sample voiced frames
        f = [float(np.nanmedian([fm.get_value_at_time(i, t) for t in voiced_times]))
             for i in (1, 2, 3, 4)]
        raw.f1_hz, raw.f2_hz, raw.f3_hz, raw.f4_hz = f
        raw.formant_dispersion_hz = (f[3] - f[0]) / 3
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 3. Behaviour vector: how the delivery differs from the neutral render
# --------------------------------------------------------------------------- #


def relative(raw: RawFeatures, neutral: RawFeatures) -> np.ndarray:
    """Six numbers, all zero for the neutral render itself. Order = FEATURE_NAMES.

    Both clips must be the same line in the same voice, so the text cancels out.
    """
    def log_ratio(a: float, b: float) -> float:
        return math.log(max(a, 1e-9) / max(b, 1e-9))

    return np.array([
        log_ratio(neutral.speech_s, raw.speech_s),                 # tempo: + is faster
        raw.pause_share - neutral.pause_share,                     # pause: + is more pauses
        12 * math.log2(raw.f0_median_hz / neutral.f0_median_hz),   # pitch level, semitones
        log_ratio(raw.f0_range_st, neutral.f0_range_st),           # pitch range
        log_ratio(raw.loud_dyn_db, neutral.loud_dyn_db),           # loudness dynamics
        log_ratio(raw.centroid_hz, neutral.centroid_hz),           # brightness
    ])


def fit_scale(relative_vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Spread of each feature across your calibration clips. Replaces DEFAULT_SCALE."""
    return np.maximum(np.vstack(list(relative_vectors)).std(axis=0), 1e-3)


def behaviour(raw: RawFeatures, neutral: RawFeatures, scale: np.ndarray = DEFAULT_SCALE) -> np.ndarray:
    """The behaviour vector. Neutral is the origin, distance is Euclidean."""
    return relative(raw, neutral) / scale


# --------------------------------------------------------------------------- #
# 4. Validity: is the clip listenable and sensible? A number from 0 to 1
# --------------------------------------------------------------------------- #


def _ramp(x: float, full: float, zero: float) -> float:
    """1 at `full`, 0 at `zero`, linear in between. Works in either direction."""
    return float(np.clip((x - zero) / (full - zero), 0.0, 1.0))


def _is_content(word: str) -> bool:
    return word.strip(PUNCT + "\"'").lower() not in STOPWORDS


def validity(raw: RawFeatures, neutral: RawFeatures, *, n_ops: int = 0,
             emphasised_words: Sequence[str] = (), pause_after_words: Sequence[str] = (),
             cfg: Config = CFG) -> tuple[float, dict[str, float]]:
    """Product of six soft checks. Returns (validity, the individual checks).

    emphasised_words:  the words the genome stressed, e.g. ["never", "gold"]
    pause_after_words: the word in front of each pause, with its punctuation, e.g. ["gold,"]
    """
    ratio = raw.duration_s / neutral.duration_s
    if ratio < cfg.dur_full[0]:
        duration = _ramp(ratio, cfg.dur_full[0], cfg.dur_zero[0])
    elif ratio > cfg.dur_full[1]:
        duration = _ramp(ratio, cfg.dur_full[1], cfg.dur_zero[1])
    else:
        duration = 1.0
    silence = (_ramp(raw.pause_share, cfg.silence_full, cfg.silence_zero)
               if raw.pause_share > cfg.silence_full else 1.0)
    voicing = 1.0 if (raw.voiced_share >= cfg.min_voiced_share
                      and not math.isnan(raw.f0_median_hz)) else 0.0
    emphasis = (sum(_is_content(w) for w in emphasised_words) / len(emphasised_words)
                if emphasised_words else 1.0)
    pauses = (sum((w.strip()[-1:] in PUNCT) or _is_content(w) for w in pause_after_words)
              / len(pause_after_words) if pause_after_words else 1.0)
    if n_ops <= cfg.parsimony_free_ops:
        parsimony = 1.0
    else:
        span = cfg.parsimony_max_ops - cfg.parsimony_free_ops
        over = min(n_ops, cfg.parsimony_max_ops) - cfg.parsimony_free_ops
        parsimony = 1.0 - (1.0 - cfg.parsimony_floor) * over / span

    checks = {"duration": duration, "silence": silence, "voicing": voicing,
              "emphasis_sense": float(emphasis), "pause_sense": float(pauses),
              "parsimony": parsimony}
    return float(np.prod(list(checks.values()))), checks


# --------------------------------------------------------------------------- #
# 5. Expressiveness: novelty through a Wundt curve
# --------------------------------------------------------------------------- #


def novelty(b: np.ndarray, archive: Sequence[np.ndarray] | None = None, k: int = CFG.knn) -> float:
    """Mean distance to the k nearest points heard before.

    With no archive this is simply the distance from the neutral render (the origin).
    """
    points = [np.zeros_like(b)] + list(archive or [])
    d = np.sort(np.linalg.norm(np.vstack(points) - b, axis=1))
    return float(d[:k].mean())


def wundt(n: float, n0: float = CFG.wundt_n0, sigma: float = CFG.wundt_sigma) -> float:
    """Bell curve with its peak at n0: boring and absurd both score low."""
    return float(math.exp(-((n - n0) ** 2) / (2 * sigma ** 2)))


# --------------------------------------------------------------------------- #
# 6. Intelligibility: word error rate (optional, needs Whisper)
# --------------------------------------------------------------------------- #

_whisper = None  # (kind, model), loaded on first use


def normalise_text(text: str) -> list[str]:
    """Lowercase words only. Delivery tags such as [breath] and <strong> are removed."""
    text = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", text)
    return re.findall(r"[a-z0-9']+", text.lower())


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


def wer_between(reference_text: str, heard_text: str) -> float:
    """Word error rate: word edits needed, divided by the number of reference words."""
    ref, hyp = normalise_text(reference_text), normalise_text(heard_text)
    if not ref:
        raise ValueError("reference text has no words")
    return _edit_distance(ref, hyp) / len(ref)


def transcribe(wav: str | Path, cfg: Config = CFG) -> str:
    """What a speech recogniser hears. Uses faster-whisper if installed, else openai-whisper."""
    global _whisper
    if _whisper is None:
        try:
            from faster_whisper import WhisperModel
            _whisper = ("faster", WhisperModel(cfg.whisper_size, device="auto", compute_type="int8"))
        except ImportError:
            import whisper  # raises ImportError if neither library is installed
            _whisper = ("openai", whisper.load_model(cfg.whisper_size))
    kind, model = _whisper
    if kind == "faster":
        segments, _ = model.transcribe(str(wav), language=cfg.language, beam_size=1)
        return " ".join(s.text for s in segments)
    return model.transcribe(str(wav), language=cfg.language, fp16=False)["text"]


def word_error_rate(wav: str | Path, text: str, cfg: Config = CFG) -> float:
    return wer_between(text, transcribe(wav, cfg))


# --------------------------------------------------------------------------- #
# 7. Same speaker: similarity of speaker embeddings (optional, needs Resemblyzer)
# --------------------------------------------------------------------------- #

_encoder = None
_embedding_cache: dict[tuple[str, float], np.ndarray] = {}


def speaker_embedding(wav: str | Path) -> np.ndarray:
    global _encoder
    from resemblyzer import VoiceEncoder, preprocess_wav
    if _encoder is None:
        _encoder = VoiceEncoder(verbose=False)
    key = (str(wav), Path(wav).stat().st_mtime)
    if key not in _embedding_cache:
        _embedding_cache[key] = _encoder.embed_utterance(preprocess_wav(Path(wav)))
    return _embedding_cache[key]


def speaker_similarity(wav: str | Path, reference_wav: str | Path) -> float:
    """Cosine similarity of the two voices, about 0 to 1. Higher = more alike."""
    a, b = speaker_embedding(wav), speaker_embedding(reference_wav)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------- #
# 8. The reward
# --------------------------------------------------------------------------- #


@dataclass
class Reward:
    score: int                      # 0 to 100, what voice_reward() returns by default
    value: float                    # the same reward as a float from 0 to 1 (no ties)
    parts: dict[str, float]         # validity, intelligibility, same_speaker, expressiveness
    metrics: dict[str, float]       # every number that was measured
    checks: dict[str, float]        # the six validity checks
    behaviour: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURE_NAMES)))
    skipped: list[str] = field(default_factory=list)  # guards that could not run


_neutral_cache: dict[tuple[str, float], RawFeatures] = {}
_warned: set[str] = set()


def _neutral_features(neutral: str | Path | RawFeatures, cfg: Config) -> RawFeatures:
    if isinstance(neutral, RawFeatures):
        return neutral
    key = (str(neutral), Path(neutral).stat().st_mtime)
    if key not in _neutral_cache:
        _neutral_cache[key] = extract_raw(neutral, cfg)
    return _neutral_cache[key]


def _optional(name: str, mode: bool | str, fn, skipped: list[str]) -> float | None:
    """Run an optional metric. In "auto" mode a failure skips it, with one warning."""
    if mode is False:
        return None
    try:
        return fn()
    except Exception as e:  # missing library, missing model weights, unreadable audio
        if mode is True:
            raise
        reason = f"{name}: {type(e).__name__}: {e}"
        skipped.append(reason)
        if name not in _warned:
            _warned.add(name)
            warnings.warn(f"{name} guard skipped and counted as passed ({type(e).__name__}: {e})")
        return None


def voice_reward(wav: str | Path, neutral: str | Path | RawFeatures, *,
                 text: str | None = None, reference_wav: str | Path | None = None,
                 archive: Sequence[np.ndarray] | None = None, scale: np.ndarray = DEFAULT_SCALE,
                 n_ops: int = 0, emphasised_words: Sequence[str] = (),
                 pause_after_words: Sequence[str] = (),
                 use_wer: bool | str = "auto", use_similarity: bool | str = "auto",
                 details: bool = False, cfg: Config = CFG) -> int | Reward:
    """Measure one clip with every automatic metric and return its reward.

    wav            the rendered clip to judge
    neutral        the same line in the same voice with no delivery changes (wav path,
                   or RawFeatures you already extracted)
    text           the words of the line. Needed for the intelligibility guard
    reference_wav  the voice the clip should still sound like. Defaults to `neutral`
    archive        behaviour vectors heard earlier in the run. Without it, novelty is
                   the distance from neutral
    scale          per-feature step sizes from fit_scale(). Defaults to rough values
    use_wer / use_similarity
                   "auto" = use when possible, skip quietly otherwise
                   True   = required, raise if it cannot run
                   False  = never run (fast mode for inside the evolution loop)
    details        False returns an int 0..100. True returns the full Reward

    reward = validity * intelligibility * same_speaker * expressiveness
    """
    skipped: list[str] = []
    try:
        neutral_raw = _neutral_features(neutral, cfg)
        raw = extract_raw(wav, cfg)
        b = behaviour(raw, neutral_raw, scale)
        if not np.all(np.isfinite(b)):
            raise ValueError("pitch could not be tracked")
    except ValueError as e:  # silent or unmeasurable clip: worst reward, never a crash
        r = Reward(0, 0.0, {"validity": 0.0}, {}, {}, skipped=[f"unmeasurable: {e}"])
        return r if details else r.score

    v, checks = validity(raw, neutral_raw, n_ops=n_ops, emphasised_words=emphasised_words,
                         pause_after_words=pause_after_words, cfg=cfg)
    n = novelty(b, archive, cfg.knn)
    expressiveness = wundt(n, cfg.wundt_n0, cfg.wundt_sigma)

    wer = None
    if text is None:
        if use_wer is True:
            raise ValueError("use_wer=True needs the text of the line")
        if use_wer == "auto":
            skipped.append("intelligibility: no text given")
    else:
        wer = _optional("intelligibility", use_wer, lambda: word_error_rate(wav, text, cfg), skipped)
    intelligibility = 1.0 if wer is None else _ramp(wer, cfg.wer_full, cfg.wer_zero)

    ref = reference_wav if reference_wav is not None else (
        neutral if not isinstance(neutral, RawFeatures) else None)
    sim = None
    if ref is None:
        if use_similarity is True:
            raise ValueError("use_similarity=True needs reference_wav or a neutral wav path")
        if use_similarity == "auto":
            skipped.append("same_speaker: no reference clip given")
    else:
        sim = _optional("same_speaker", use_similarity, lambda: speaker_similarity(wav, ref), skipped)
    same_speaker = 1.0 if sim is None else _ramp(sim, cfg.sim_full, cfg.sim_zero)

    value = v * intelligibility * same_speaker * expressiveness
    metrics = {**raw.to_dict(), "novelty": n,
               "wer": math.nan if wer is None else wer,
               "speaker_similarity": math.nan if sim is None else sim}
    r = Reward(score=int(round(100 * value)), value=float(value),
               parts={"validity": v, "intelligibility": intelligibility,
                      "same_speaker": same_speaker, "expressiveness": expressiveness},
               metrics=metrics, checks=checks, behaviour=b, skipped=skipped)
    return r if details else r.score


# --------------------------------------------------------------------------- #
# 9. Logbook: one JSON line per evaluation. plot_measurements.py reads this
# --------------------------------------------------------------------------- #


def make_record(reward: Reward, *, run: str, line_id: int, character: str, generation: int,
                individual: int, genome: str, wav: str = "") -> dict:
    """Flatten one evaluation into a single row with plain column names."""
    rec = {"run": run, "line_id": line_id, "character": character, "generation": generation,
           "individual": individual, "genome": genome, "wav": wav,
           "fitness": reward.value, "reward": reward.score,
           "novelty": reward.metrics.get("novelty"), "wundt": reward.parts.get("expressiveness"),
           **reward.parts}
    rec.update({f"b_{name}": float(x) for name, x in zip(FEATURE_NAMES, reward.behaviour)})
    rec.update({f"raw_{k}": v for k, v in reward.metrics.items()})
    rec.update({f"check_{k}": v for k, v in reward.checks.items()})
    return rec


def log_evaluation(path: str | Path, record: dict) -> None:
    """Append one record to the logbook."""
    clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in record.items()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean) + "\n")


# --------------------------------------------------------------------------- #
# 10. Command line
# --------------------------------------------------------------------------- #


def measure_folder(folder: str | Path, out: str | Path, neutral_wav: str | Path | None = None,
                   cfg: Config = CFG) -> list[dict]:
    """Acoustic features of every wav in a folder, as a CSV (voice map, calibration)."""
    wavs = sorted(Path(folder).glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav files in {folder}")
    neutral = extract_raw(neutral_wav, cfg) if neutral_wav else None
    rows = []
    for w in wavs:
        try:
            raw = extract_raw(w, cfg)
        except ValueError as e:
            print(f"skipped {w.name}: {e}")
            continue
        row = {"file": w.name, **raw.to_dict()}
        if neutral is not None:
            row.update({f"rel_{n}": float(x) for n, x in zip(FEATURE_NAMES, relative(raw, neutral))})
        rows.append(row)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"measured {len(rows)} clips -> {out}")
    return rows


def _print_reward(r: Reward) -> None:
    print(f"\nreward: {r.score} / 100   (value {r.value:.3f})\n")
    for name, x in r.parts.items():
        print(f"  {name:16s} {x:.2f}")
    print("\n  validity checks: " + ", ".join(f"{k} {v:.2f}" for k, v in r.checks.items()))
    print("  behaviour:       " + ", ".join(f"{k} {v:+.2f}" for k, v in zip(FEATURE_NAMES, r.behaviour)))
    m = r.metrics
    if m:
        print(f"  novelty {m['novelty']:.2f}, WER {m['wer']:.2f}, speaker similarity {m['speaker_similarity']:.2f}")
    for s in r.skipped:
        print(f"  skipped -> {s}")


def main() -> None:
    p = argparse.ArgumentParser(description="Automatic voice metrics and reward.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reward", help="reward of one clip, with the full breakdown")
    r.add_argument("wav")
    r.add_argument("--neutral", required=True, help="neutral render of the same line")
    r.add_argument("--text", help="the words of the line (turns on the intelligibility guard)")
    r.add_argument("--reference", help="voice the clip should still sound like (default: neutral)")
    m = sub.add_parser("measure", help="acoustic features of every wav in a folder, as CSV")
    m.add_argument("folder")
    m.add_argument("--out", default="measurements.csv")
    m.add_argument("--neutral", help="neutral clip of the same line, adds the relative features")
    args = p.parse_args()
    if args.cmd == "reward":
        _print_reward(voice_reward(args.wav, args.neutral, text=args.text,
                                   reference_wav=args.reference, details=True))
    else:
        measure_folder(args.folder, args.out, args.neutral)


if __name__ == "__main__":
    main()
