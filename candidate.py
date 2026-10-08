import re
from copy import deepcopy


class Candidate:
    def __init__(self, operations=None):
        self.operations = operations or []

    def copy(self):
        return Candidate(deepcopy(self.operations))

    def resolve(self, text):
        """
        Convert the abstract genome into operations understood
        by tts_formatter.py.
        """
        words = re.findall(r"[A-Za-z']+", text)

        if not words:
            raise ValueError("Text contains no words.")

        resolved = []

        for gene in self.operations:
            feature = gene["feature"]

            if "position" in gene:
                position = max(
                    0.0,
                    min(1.0, gene["position"])
                )

                index = round(
                    position * (len(words) - 1)
                )

                kwargs = {
                    "target": words[index]
                }

                if "kind" in gene:
                    kwargs["pause_type"] = gene["kind"]

                resolved.append({
                    "feature": feature,
                    "kwargs": kwargs
                })

            elif feature == "speed":
                resolved.append({
                    "feature": "speed",
                    "kwargs": {
                        "rate": gene["rate"]
                    }
                })

            else:
                raise ValueError(
                    f"Cannot resolve gene: {gene}"
                )

        return resolved

    def __repr__(self):
        return f"Candidate(operations={self.operations})"