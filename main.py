from pathlib import Path
import sys

COSYVOICE_ROOT = Path("CosyVoice").resolve()
MATCHA_ROOT = COSYVOICE_ROOT / "third_party" / "Matcha-TTS"

sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(MATCHA_ROOT))

from cosyvoice.cli.cosyvoice import CosyVoice2

from genome_factory import GenomeFactory
from genome_history import GenomeHistory
from explorer import Explorer
from dramatizer import Dramatizer, DramatizerPipeline


MODEL_DIR = (
    COSYVOICE_ROOT /
    "pretrained_models" /
    "CosyVoice2-0.5B"
)

VOICE_DATA = Path("Voices/Voices")

VOICE_ID = "ahmed"

reference_audio = (
    VOICE_DATA /
    VOICE_ID /
    "reference.wav"
)

reference_text = (
    VOICE_DATA /
    VOICE_ID /
    "reference.txt"
).read_text(
    encoding="utf-8"
).strip()

print("Loading CosyVoice...")

cosyvoice = CosyVoice2(
    str(MODEL_DIR),
    load_jit=False,
    load_trt=False,
    load_vllm=False,
    fp16=False,
)

print("CosyVoice loaded.")

history = GenomeHistory(
    "history/voices.json"
)

factory = GenomeFactory(
    min_operations=1,
    max_operations=4
)

pipeline = DramatizerPipeline(
    cosyvoice=cosyvoice
)

explorer = Explorer(
    genome_factory=factory,
    pipeline=pipeline,
    output_dir="output/exploration",
    max_wer=0.10,
    min_similarity=0.75,
)

dramatizer = Dramatizer(
    history=history,
    genome_factory=factory,
    pipeline=pipeline,
    explorer=explorer,
)

results = dramatizer.run(
    voice_id=VOICE_ID,
    reference_audio=reference_audio,
    reference_text=reference_text,
    use_history=True,
    steps=30,
    n_outputs=10,
)

print(f"\nSelected {len(results)} candidates:")

for i, result in enumerate(results, start=1):
    print(
        f"{i}. {result.audio_path} "
        f"(WER={result.wer:.3f}, "
        f"SIM={result.speaker_similarity:.3f})"
    )