import re


def tokenize(text):
    return re.findall(r"[A-Za-z']+", text)


def migrate_genome(genome, reference_text):
    """
    Convert old tts_formatter-style operations into
    the new abstract genome representation.

    Already-migrated genes are preserved.
    """
    words = tokenize(reference_text)

    migrated = []

    for gene in genome:
        feature = gene["feature"]

        # Already in the new format
        if "kwargs" not in gene:
            migrated.append(gene.copy())
            continue

        kwargs = gene["kwargs"]

        # Speed
        if feature == "speed":
            migrated.append({
                "feature": "speed",
                "rate": float(kwargs["rate"])
            })
            continue

        # Position-based operations
        if "target" in kwargs:
            target = str(kwargs["target"]).lower()

            matching_indices = [
                i
                for i, word in enumerate(words)
                if word.lower() == target
            ]

            if not matching_indices:
                raise ValueError(
                    f"Cannot migrate {feature}: "
                    f"target {target!r} not found in "
                    f"historical reference text."
                )

            # Old format doesn't always identify
            # which repeated occurrence was targeted.
            occurrence = int(kwargs.get("occurrence", 0))

            if not 0 <= occurrence < len(matching_indices):
                raise ValueError(
                    f"Invalid occurrence for {target!r}"
                )

            index = matching_indices[occurrence]

            position = (
                index / (len(words) - 1)
                if len(words) > 1
                else 0.0
            )

            new_gene = {
                "feature": feature,
                "position": round(position, 6)
            }

            # Preserve pause style where possible
            if feature == "pause":
                if "style" in kwargs:
                    new_gene["kind"] = kwargs["style"]

            migrated.append(new_gene)
            continue

        raise ValueError(
            f"Unsupported historical gene: {gene}"
        )

    return migrated
