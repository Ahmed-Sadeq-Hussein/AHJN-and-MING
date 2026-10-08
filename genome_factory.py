import random
import re

from candidate import Candidate


class GenomeFactory:
    def __init__(self, min_operations=1, max_operations=5):
        self.min_operations = min_operations
        self.max_operations = max_operations

        self.targeted_features = [
            "emphasis",
            "breath",
            "quick_breath",
            "noise",
            "laughter",
            "cough",
            "clucking",
            "accent",
            "hissing",
            "sigh",
            "vocalized_noise",
            "lipsmack",
            "mn",
            "laughing_speech",
        ]

        self.features = (
            self.targeted_features
            + ["pause", "speed"]
        )


    def random_genome(self, text):
        """Create a completely random candidate."""

        n_operations = random.randint(
            self.min_operations,
            self.max_operations
        )

        operations = [
            self._random_operation(text)
            for _ in range(n_operations)
        ]

        return Candidate(operations)


    def mutate(self, candidate, text):
        child = candidate.copy()

        possible_mutations = ["add"]

        if child.operations:
            possible_mutations += [
                "remove",
                "modify",
                "replace",
            ]

        mutation = random.choice(possible_mutations)

        if mutation == "add":
            self._add_operation(child, text)

        elif mutation == "remove":
            self._remove_operation(child)

        elif mutation == "modify":
            self._modify_operation(child, text)

        elif mutation == "replace":
            self._replace_operation(child, text)

        return child, mutation


    def _add_operation(self, candidate, text):
        if len(candidate.operations) >= self.max_operations:
            self._modify_operation(candidate, text)
            return

        candidate.operations.append(
            self._random_operation(text)
        )

    def _remove_operation(self, candidate):
        if len(candidate.operations) <= self.min_operations:
            return

        index = random.randrange(len(candidate.operations))
        candidate.operations.pop(index)

    def _modify_operation(self, candidate, text):
        if not candidate.operations:
            self._add_operation(candidate, text)
            return

        index = random.randrange(len(candidate.operations))
        gene = candidate.operations[index]

        # Move a targeted operation slightly through the sentence.
        if "position" in gene:
            shift = random.uniform(-0.15, 0.15)

            gene["position"] = max(
                0.0,
                min(1.0, gene["position"] + shift)
            )

        # Change speaking speed slightly.
        elif gene["feature"] == "speed":
            shift = random.uniform(-0.1, 0.1)

            gene["rate"] = round(
                max(
                    0.5,
                    min(2.0, gene["rate"] + shift)
                ),
                2
            )

    def _replace_operation(self, candidate, text):
        if not candidate.operations:
            self._add_operation(candidate, text)
            return

        index = random.randrange(len(candidate.operations))

        candidate.operations[index] = (
            self._random_operation(text)
        )


    def _random_operation(self, text):
        feature = random.choice(self.features)

        return self._random_operation_for_feature(
            feature,
            text
        )


    def _random_operation_for_feature(self, feature, text):

        # All features that target a position in the sentence
        if feature in self.targeted_features:
            return {
                "feature": feature,
                "position": round(random.random(), 3)
            }

        if feature == "pause":
            return {
                "feature": "pause",
                "position": round(random.random(), 3),
                "kind": random.choice([
                    "dramatic",
                    "realization"
                ])
            }

        if feature == "speed":
            return {
                "feature": "speed",
                "rate": round(
                    random.uniform(0.7, 1.3),
                    2
                )
            }

        raise ValueError(
            f"Unsupported feature: {feature}"
        )


    def _words(self, text):
        words = re.findall(r"[A-Za-z']+", text)

        if not words:
            raise ValueError(
                "Cannot generate targeted operations: "
                "text contains no words."
            )

        return words