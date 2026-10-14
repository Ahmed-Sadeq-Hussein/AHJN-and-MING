
"""
Generate speech using one voice and a genome inherited
from another voice in history/voices.json.

Example:
    python generate_from_history.py --voice ahmed --genome jhin
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torchaudio


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

COSYVOICE_ROOT = PROJECT_ROOT / "CosyVoice"
MATCHA_ROOT = COSYVOICE_ROOT / "third_party" / "Matcha-TTS"

MODEL_DIR = (
    COSYVOICE_ROOT
    / "pretrained_models"
    / "CosyVoice2-0.5B"
)

VOICE_DIR = PROJECT_ROOT / "Voices" / "Voices"
HISTORY_FILE = PROJECT_ROOT / "history" / "voices.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "generated"

# Add CosyVoice dependencies to the Python path.
sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(MATCHA_ROOT))

from cosyvoice.cli.cosyvoice import CosyVoice2

from candidate import Candidate
from tts_formatter import process_tts_string, synthesize


# --------------------------------------------------
# Sentence
# --------------------------------------------------

DEFAULT_TEXT = (
    "The Dramatizer is a well made machine "
    "and it made me into the man I am"
)


# --------------------------------------------------
# Load a historical genome
# --------------------------------------------------

def load_genome(voice_id):
    if not HISTORY_FILE.is_file():
        raise FileNotFoundError(
            f"History file not found: {HISTORY_FILE}"
        )

    with HISTORY_FILE.open("r", encoding="utf-8") as f:
        records = json.load(f)

    matches = [
        record
        for record in records
        if record.get("voice_id", "").lower() == voice_id.lower()
    ]

    if not matches:
        available = [
            record.get("voice_id", "?")
            for record in records
        ]

        raise ValueError(
            f"No genome found for '{voice_id}'. "
            f"Available genomes: {available}"
        )

    # If multiple records exist, prefer the highest MOS.
    record = max(
        matches,
        key=lambda r: float(r.get("mos", 0))
    )

    return Candidate(record["genome"]), record


# --------------------------------------------------
# Generate speech
# --------------------------------------------------

def generate(voice_id, genome_id, text):
    reference_audio = (
        VOICE_DIR / voice_id / "reference.wav"
    )

    reference_text_file = (
        VOICE_DIR / voice_id / "reference.txt"
    )

    if not reference_audio.is_file():
        raise FileNotFoundError(
            f"Reference audio not found: {reference_audio}"
        )

    if not reference_text_file.is_file():
        raise FileNotFoundError(
            f"Reference transcript not found: {reference_text_file}"
        )

    reference_text = reference_text_file.read_text(
        encoding="utf-8"
    ).strip()

    if not reference_text:
        raise ValueError("Reference transcript is empty.")

    # Load inherited genome.
    candidate, record = load_genome(genome_id)

    print(f"\nTarget voice: {voice_id}")
    print(f"Genome source: {record['voice_id']}")
    print(f"Historical MOS: {record.get('mos', 'N/A')}")
    print(f"Sentence: {text}")

    print("\nInherited genome:")
    for gene in candidate.operations:
        print(f"  {gene}")

    # Convert normalized positions into word targets.
    operations = candidate.resolve(text)

    print("\nResolved operations:")
    for operation in operations:
        print(f"  {operation}")

    # Build the TTS plan.
    plan = process_tts_string(
        text,
        operations
    )

    print("\nFormatted text:")
    print(plan.text)

    # Load CosyVoice2.
    print("\nLoading CosyVoice2...")

    cosyvoice = CosyVoice2(
        str(MODEL_DIR),
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False,
    )

    print("Generating speech...")

    with torch.inference_mode():
        audio = synthesize(
            cosyvoice=cosyvoice,
            plan=plan,
            prompt_wav=str(reference_audio),
            prompt_text=reference_text,
        )

    # Save the generated audio.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    def safe_name(value):
        return re.sub(
            r"[^A-Za-z0-9_-]+", "_", value
        ).strip("_")

    output_path = OUTPUT_DIR / (
        f"{safe_name(voice_id)}"
        f"_genome_{safe_name(genome_id)}.wav"
    )

    torchaudio.save(
        str(output_path),
        audio.detach().cpu(),
        cosyvoice.sample_rate
    )

    print(f"\nGenerated audio saved to: {output_path}")

    return output_path


# --------------------------------------------------
# Command-line interface
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate speech using a target voice "
            "and a historical Dramatizer genome."
        )
    )

    parser.add_argument(
        "--voice",
        required=True,
        help="Target voice ID, e.g. ahmed"
    )

    parser.add_argument(
        "--genome",
        required=True,
        help="Historical genome ID, e.g. jhin"
    )

    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Sentence to synthesize"
    )

    args = parser.parse_args()

    generate(
        voice_id=args.voice,
        genome_id=args.genome,
        text=args.text,
    )
