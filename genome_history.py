import json
from pathlib import Path

from candidate import Candidate
from genome_migration import migrate_genome
from unsupervised_measurements import speaker_similarity


class GenomeHistory:
    def __init__(self, history_path="history/voices.json"):
        self.history_path = Path(history_path)
        self.records = []
        self.load()



    def load(self):
        if not self.history_path.exists():
            self.records = []
            return

        with self.history_path.open("r", encoding="utf-8") as f:
            self.records = json.load(f)

        changed = False

        for record in self.records:
            genome = record["genome"]

            # Only migrate old-format genomes
            if not any("kwargs" in gene for gene in genome):
                continue

            reference_audio = Path(record["reference_audio"])
            reference_text_path = reference_audio.with_name(
                "reference.txt"
            )

            if not reference_text_path.exists():
                raise FileNotFoundError(
                    "Cannot migrate historical genome: "
                    f"{reference_text_path} does not exist."
                )

            reference_text = reference_text_path.read_text(
                encoding="utf-8"
            ).strip()

            record["genome"] = migrate_genome(
                genome,
                reference_text
            )

            changed = True

        if changed:
            # Preserve the original history before migration
            backup_path = self.history_path.with_suffix(
                ".json.bak"
            )

            if not backup_path.exists():
                import shutil
                shutil.copy2(self.history_path, backup_path)

            self.save()
            print("Historical genomes migrated successfully.")


    def save(self):
        """Save all history records to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        with self.history_path.open("w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=4)


    def find_nearest_voice(
        self,
        reference_audio,
        exclude_voice_id=None
    ):
        if not self.records:
            return None

        best_record = None
        best_similarity = -1.0

        for record in self.records:

            if (
                exclude_voice_id is not None
                and record["voice_id"] == exclude_voice_id
            ):
                continue

            stored_audio = Path(
                record["reference_audio"]
            )

            if not stored_audio.exists():
                continue

            similarity = speaker_similarity(
                reference_audio,
                stored_audio
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_record = record

        if best_record is None:
            return None

        return best_record, best_similarity


    def get_starting_genome(self, reference_audio):
        """
        Return a Candidate copied from the nearest previous voice.
        Returns None if no usable history exists.
        """
        result = self.find_nearest_voice(reference_audio)

        if result is None:
            return None

        record, similarity = result

        return Candidate(record["genome"]), similarity


    def store_winner(self, voice_id, reference_audio, candidate, mos, wer, speaker_similarity_score):
        """Store the human-selected winning genome."""

        record = {
            "schema_version": 2,
            "voice_id": voice_id,
            "reference_audio": str(reference_audio),
            "genome": candidate.operations,
            "mos": float(mos),
            "wer": float(wer),
            "speaker_similarity": float(speaker_similarity_score)
        }

        self.records.append(record)
        self.save()