import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class ExplorationLogger:
    def __init__(
        self,
        output_dir,
        voice_id,
        reference_audio,
        reference_text,
        start_candidate,
        steps,
        max_wer,
        min_similarity,
    ):
        self.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid4().hex[:6]
        )

        self.run_dir = Path(output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.record = {
            "run_id": self.run_id,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": "running",
            "voice_id": voice_id,
            "reference_audio": str(
                Path(reference_audio).resolve()
            ),
            "reference_text": reference_text,
            "configuration": {
                "steps": steps,
                "max_wer": max_wer,
                "min_similarity": min_similarity,
            },
            "starting_genome": start_candidate.operations,
            "candidates": [],
        }

        self.save()

    def audio_path(self, step):
        return self.run_dir / f"candidate_{step:03d}.wav"

    def add_candidate(
        self,
        step,
        parent_step,
        candidate,
        mutation,
        status,
        audio_path=None,
        reward=None,
        error=None,
    ):
        entry = {
            "step": step,
            "parent_step": parent_step,
            "genome": candidate.operations,
            "mutation": mutation,
            "status": status,
            "audio_path": (
                Path(audio_path).name
                if audio_path is not None
                else None
            ),
            "score": None,
            "wer": None,
            "speaker_similarity": None,
            "metrics": {},
            "parts": {},
            "skipped": [],
            "error": error,
        }

        if reward is not None:
            entry["score"] = reward.score
            entry["wer"] = reward.metrics.get("wer")
            entry["speaker_similarity"] = (
                reward.metrics.get("speaker_similarity")
            )
            entry["metrics"] = reward.metrics
            entry["parts"] = reward.parts
            entry["skipped"] = reward.skipped

        self.record["candidates"].append(entry)
        self.save()

    def finish(self):
        self.record["status"] = "completed"
        self.record["finished_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        self.save()

    def save(self):
        def clean(value):
            """Convert NumPy values and non-finite numbers to JSON."""
            if hasattr(value, "item"):
                value = value.item()

            if isinstance(value, Path):
                return str(value)

            if isinstance(value, float):
                return value if math.isfinite(value) else None

            if isinstance(value, dict):
                return {
                    str(k): clean(v)
                    for k, v in value.items()
                }

            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]

            return value

        path = self.run_dir / "run.json"
        temp_path = self.run_dir / "run.json.tmp"

        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(
                clean(self.record),
                f,
                indent=4,
                ensure_ascii=False,
            )

        temp_path.replace(path)
