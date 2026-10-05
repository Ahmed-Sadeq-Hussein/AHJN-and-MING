from tts_formatter import process_tts_string


class DramatizerPipeline:

    def __init__(self, generator, evaluator):
        self.generator = generator
        self.evaluator = evaluator


    def evaluate_candidate(self, candidate, reference_text, reference_audio, output_path):
        plan = process_tts_string(reference_text, candidate.operations)

        self.generator.generate(plan=plan, reference_audio=reference_audio, reference_text=reference_text, output_path=output_path)

        scores = self.evaluator.evaluate(reference_audio=reference_audio, generated_audio=output_path, reference_text=reference_text, candidate=candidate)

        return plan, scores