from pathlib import Path

import torchaudio

from candidate import Candidate
from tts_formatter import process_tts_string, synthesize
from unsupervised_measurements import voice_reward


class DramatizerPipeline:
    def __init__(self, cosyvoice):
        self.cosyvoice = cosyvoice

    def evaluate_candidate(
        self,
        candidate,
        reference_text,
        reference_audio,
        output_path,
    ):
        # Convert abstract genome -> concrete TTS operations
        operations = candidate.resolve(reference_text)

        # Concrete operations -> TTS plan
        plan = process_tts_string(
            reference_text,
            operations
        )

        # Generate audio
        audio = synthesize(
            cosyvoice=self.cosyvoice,
            plan=plan,
            prompt_wav=str(reference_audio),
            prompt_text=reference_text
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torchaudio.save(
            output_path,
            audio.cpu(),
            self.cosyvoice.sample_rate
        )

        # Evaluate generated audio
        reward = voice_reward(
            output_path,
            reference_audio,
            text=reference_text,
            details=True,
            use_wer=True
        )

        return plan.text, reward


class Dramatizer:
    def __init__(
        self,
        history,
        genome_factory,
        pipeline,
        explorer
    ):
        self.history = history
        self.genome_factory = genome_factory
        self.pipeline = pipeline
        self.explorer = explorer

    def run(
        self,
        voice_id,
        reference_audio,
        reference_text,
        use_history=True,
        n_outputs=10,
        steps=30
    ):
        if use_history:
            nearest = self.history.find_nearest_voice(
                reference_audio,
                exclude_voice_id=voice_id
            )
        else:
            nearest = None

        if nearest is not None:
            record, similarity = nearest

            print(
                f"Nearest previous voice: "
                f"{record['voice_id']} "
                f"(similarity={similarity:.3f})"
            )

            start = Candidate(record["genome"])

        else:
            start = self.genome_factory.random_genome(
                reference_text
            )

        candidates = self.explorer.explore(
            start=start,
            reference_audio=reference_audio,
            reference_text=reference_text,
            steps=steps,
            voice_id=voice_id,
        )

        selected = self.explorer.select_diverse(
            candidates,
            n=n_outputs
        )

        return selected