from pathlib import Path
import sys

COSYVOICE_ROOT = Path("CosyVoice").resolve()
MATCHA_ROOT = COSYVOICE_ROOT / "third_party" / "Matcha-TTS"

sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(MATCHA_ROOT))

import torchaudio

from candidate import Candidate
from tts_formatter import process_tts_string, synthesize
from cosyvoice.cli.cosyvoice import CosyVoice2

from unsupervised_measurements import voice_reward

COSYVOICE_ROOT = Path("CosyVoice")
MODEL_DIR = COSYVOICE_ROOT / "pretrained_models" / "CosyVoice2-0.5B"
OUTPUT_PATH = Path("output/test_candidate.wav")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

VOICE_DATA = Path("Voices/Voices")

reference_audio = VOICE_DATA / "julius_gold" / "reference.wav"
reference_text_path = VOICE_DATA / "julius_gold" / "reference.txt"
reference_text = reference_text_path.read_text(encoding="utf-8").strip()

candidate = Candidate([
    {
        "feature": "emphasis",
        "kwargs": {
            "target": "I"
        }
    },
    {
        "feature": "breath",
        "kwargs": {
            "target": "gold"
        }
    },
    {
        "feature": "speed",
        "kwargs": {
            "rate": 0.8
        }
    }
])

plan = process_tts_string(reference_text, candidate.operations)

cosyvoice = CosyVoice2(
    str(MODEL_DIR),
    load_jit=False,
    load_trt=False,
    load_vllm=False,
    fp16=False,
)

audio = synthesize(cosyvoice=cosyvoice, plan=plan, prompt_wav=str(reference_audio), prompt_text=reference_text)

print("Original text:")
print(reference_text)

print("\nCandidate:")
print(candidate.operations)

print("\nTTS Plan:")
print("Text:", plan.text)
print("Speed:", plan.speed)
print("Instruction:", plan.instruct)
print("Warnings:", plan.warnings)

torchaudio.save(
    OUTPUT_PATH,
    audio.cpu(),
    cosyvoice.sample_rate
)

reward = voice_reward(OUTPUT_PATH, reference_audio, text=reference_text, details=True, use_wer=True)

print("Score:", reward.score)
print("Value:", reward.value)
print("Parts:", reward.parts)
print("Metrics:", reward.metrics)
print("Skipped:", reward.skipped)