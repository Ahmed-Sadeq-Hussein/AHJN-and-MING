class DramatizerPipeline:

    def __init__(self, transformations, generator, evaluator):
        self.transformations = transformations
        self.generator = generator
        self.evaluator = evaluator


    def evaluate_candidate(self, candidate, reference_text, reference_audio, output_path):
        modified_text = candidate.apply(reference_text, self.transformations)

        self.generator.generate(text=modified_text, reference_audio=reference_audio, output_path=output_path)

        scores = self.evaluator.evaluate(reference_audio=reference_audio, generated_audio=output_path, reference_text=reference_text)

        return modified_text, scores