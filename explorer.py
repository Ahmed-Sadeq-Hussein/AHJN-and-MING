from dataclasses import dataclass
from pathlib import Path
import numpy as np
import math
import random
import json

from candidate import Candidate
from exploration_logger import ExplorationLogger


@dataclass
class ExplorationResult:
    candidate: Candidate
    audio_path: Path
    wer: float
    speaker_similarity: float
    score: int
    mutation: str
    step: int
    parent_step: int
    reward: object


class Explorer:
    def __init__(self, genome_factory, pipeline, output_dir="output/exploration", max_wer=0.10, min_similarity=0.75):
        self.genome_factory = genome_factory
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)

        self.max_wer = max_wer
        self.min_similarity = min_similarity

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _evaluate(self, candidate, reference_text, reference_audio, step, mutation, parent_step, logger,):
        output_path = logger.audio_path(step)

        modified_text, reward = (
            self.pipeline.evaluate_candidate(
                candidate=candidate,
                reference_text=reference_text,
                reference_audio=reference_audio,
                output_path=output_path,
            )
        )

        return ExplorationResult(
            candidate=candidate,
            audio_path=output_path,
            wer=reward.metrics.get("wer", float("nan")),
            speaker_similarity=reward.metrics.get(
                "speaker_similarity", float("nan")
            ),
            score=reward.score,
            mutation=mutation,
            step=step,
            parent_step=parent_step,
            reward=reward,
        )


    def _is_valid(self, result):
        if not (
            isinstance(result.wer, (int, float))
            and isinstance(result.speaker_similarity, (int, float))
        ):
            return False

        return (
            math.isfinite(result.wer)
            and math.isfinite(result.speaker_similarity)
            and result.wer <= self.max_wer
            and result.speaker_similarity >= self.min_similarity
        )


    def _genome_key(self, candidate):
        return json.dumps(
            candidate.operations,
            sort_keys=True,
            separators=(",", ":"),
        )


    def _behavior_vector(self, result):
        """
        Extract acoustic features from an evaluated candidate.
        """
        metrics = result.reward.metrics

        feature_names = [
            "speech_s",
            "pause_share",
            "f0_median_hz",
            "f0_range_st",
            "loud_dyn_db",
        ]

        return np.array(
            [
                float(metrics.get(name, float("nan")))
                for name in feature_names
            ],
            dtype=float
        )


    def select_diverse(self, candidates, n=10):
        """
        Select up to n acoustically diverse candidates using
        greedy farthest-point sampling.
        """
        if not candidates:
            return []

        if len(candidates) <= n:
            return candidates.copy()

        # Extract feature vectors.
        vectors = np.array([
            self._behavior_vector(candidate)
            for candidate in candidates
        ])

        # Replace non-finite values with per-feature medians.
        for column in range(vectors.shape[1]):
            values = vectors[:, column]
            finite = np.isfinite(values)

            median = (
                np.median(values[finite])
                if finite.any()
                else 0.0
            )

            values[~finite] = median

        # Normalize each feature to comparable scale.
        mean = vectors.mean(axis=0)
        std = vectors.std(axis=0)

        std[std < 1e-8] = 1.0

        normalized = (vectors - mean) / std

        # Begin with the candidate having the highest
        # automatic score.
        first = max(
            range(len(candidates)),
            key=lambda i: candidates[i].score
        )

        selected_indices = [first]
        remaining = set(range(len(candidates))) - {first}

        # Repeatedly choose the candidate furthest from
        # its nearest already-selected candidate.
        while len(selected_indices) < n:

            best_index = None
            best_distance = -1.0

            for index in sorted(remaining):

                min_distance = min(
                    np.linalg.norm(
                        normalized[index] - normalized[chosen]
                    )
                    for chosen in selected_indices
                )

                if min_distance > best_distance:
                    best_distance = min_distance
                    best_index = index

            selected_indices.append(best_index)
            remaining.remove(best_index)

        return [
            candidates[index]
            for index in selected_indices
        ]


    def explore(
        self,
        start,
        reference_text,
        reference_audio,
        steps=30,
        voice_id="unknown",
    ):
        logger = ExplorationLogger(
            output_dir=self.output_dir,
            voice_id=voice_id,
            reference_audio=reference_audio,
            reference_text=reference_text,
            start_candidate=start,
            steps=steps,
            max_wer=self.max_wer,
            min_similarity=self.min_similarity,
        )

        print(f"\nRun ID: {logger.run_id}")
        print(f"Output directory: {logger.run_dir}")

        seen = {self._genome_key(start)}

        # (candidate, originating step)
        parents = [(start.copy(), 0)]

        valid_results = []

        for step in range(1, steps + 1):

            parent, parent_step = random.choice(parents)

            child, mutation = self.genome_factory.mutate(
                parent,
                reference_text,
            )

            print(f"\n--- Step {step}/{steps} ---")
            print(f"Parent: {parent_step}")
            print(f"Mutation: {mutation}")
            print(f"Genome: {child.operations}")

            key = self._genome_key(child)

            if key in seen:
                print("DUPLICATE")

                logger.add_candidate(
                    step=step,
                    parent_step=parent_step,
                    candidate=child,
                    mutation=mutation,
                    status="duplicate",
                )
                continue

            seen.add(key)

            try:
                result = self._evaluate(
                    candidate=child,
                    reference_text=reference_text,
                    reference_audio=reference_audio,
                    step=step,
                    mutation=mutation,
                    parent_step=parent_step,
                    logger=logger,
                )

            except Exception as e:
                print(
                    f"Evaluation failed: "
                    f"{type(e).__name__}: {e}"
                )

                logger.add_candidate(
                    step=step,
                    parent_step=parent_step,
                    candidate=child,
                    mutation=mutation,
                    status="error",
                    audio_path=(
                        logger.audio_path(step)
                        if logger.audio_path(step).exists()
                        else None
                    ),
                    error=f"{type(e).__name__}: {e}",
                )
                continue

            is_valid = self._is_valid(result)

            status = "accepted" if is_valid else "rejected"

            logger.add_candidate(
                step=step,
                parent_step=parent_step,
                candidate=child,
                mutation=mutation,
                status=status,
                audio_path=result.audio_path,
                reward=result.reward,
            )

            print(f"WER: {result.wer:.3f}")
            print(f"Similarity: {result.speaker_similarity:.3f}")
            print(status.upper())

            if is_valid:
                valid_results.append(result)
                parents.append((child, step))

        logger.finish()

        print(f"\nRun saved to: {logger.run_dir}")
        print(f"Valid candidates: {len(valid_results)}")

        return valid_results