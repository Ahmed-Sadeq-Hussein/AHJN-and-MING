from tts_formatter import process_tts_string
from candidate import Candidate


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
        reference_audio,
        reference_text,
        use_history=True,
        n_outputs=10
    ):
        # 1. Starting genome
        if use_history:
            nearest = self.history.find_nearest(reference_audio)
        else:
            nearest = None

        if nearest is not None:
            start = Candidate(nearest["genome"])
        else:
            start = self.genome_factory.random_genome(
                reference_text
            )

        # 2. Explore nearby genomes
        candidates = self.explorer.explore(
            start=start,
            reference_audio=reference_audio,
            reference_text=reference_text,
        )

        # 3. Select diverse valid examples
        selected = self.explorer.select_diverse(
            candidates,
            n=n_outputs
        )

        return selected