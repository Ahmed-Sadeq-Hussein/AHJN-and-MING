"""Local three-rater audio evaluation for Dramatizer.

Run: streamlit run mos_evaluator.py
Install: python -m pip install streamlit pandas
"""
from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import math
from candidate import Candidate
from genome_history import GenomeHistory

import pandas as pd
import streamlit as st

# Candidate numbers are exploration STEP IDs, not ordinal indices.
RUNS = {
    "20261012_142127_Ego": [18, 60, 67, 24, 36, 2, 3, 97, 93],
    "20261012_144336_evelynn": [43, 17, 36, 34, 94, 70, 40, 41, 88, 68],
    "20261012_145820_hugo": [16, 22, 63, 65, 92],
    "20261012_150555_jhin": [95, 59, 58, 77, 91, 7, 74, 85, 82, 18],
    "20261012_151726_Joffrey": [2, 89, 94, 65, 54, 81, 28, 84, 90, 16],
    "20261012_152931_julius": [93, 53, 84, 67, 90, 97, 40, 25, 45],
    "20261012_154156_julius_gold": [84, 68, 65, 97, 57, 42, 53, 10, 41, 88],
    "20261012_154847_mitchell": [1, 12, 51, 57, 63, 74, 77, 85, 92],
    "20261012_172736_mordekaiser": [36, 97, 69, 42, 59, 67, 100, 9, 7, 3],
    "20261012_184636_riven": [18, 78, 95, 63, 39, 81, 77, 76, 98, 48],
    "20261012_200428_samira": [60, 54, 100, 18, 23, 66, 38, 99, 36, 24],
    "20261012_202108_Silco": [97, 80, 24, 7, 70, 32, 9, 34, 81, 96],
    "20261012_210211_swain": [20, 65, 13, 86, 47, 62, 70, 66, 28, 11],
    "20261012_211900_tahmkench": [35, 91, 34, 38, 32, 6, 2, 19, 28, 54],
    "20261012_221918_thresh": [39, 99, 95, 65, 37, 66, 41, 52, 48, 92],
    "20261012_223129_Tywin": [61, 20, 91, 57, 29, 14, 25, 37, 40, 89],
    "20261014_134613_ahmed": [85, 57, 61, 75, 47, 13, 23, 96, 42, 34],
    "20261014_135718_Tywin_2": [44, 87, 43, 52, 41, 47, 7, 93, 76, 26]
}

NUM_EVALUATORS = 2

DEFAULT_QUESTIONS = [
    "How good is this voice?",
]

NUM_QUESTIONS = len(DEFAULT_QUESTIONS)

st.set_page_config(page_title="Dramatizer MOS evaluation", page_icon="🎧", layout="wide")
st.title("🎧 Dramatizer — listening evaluation")
st.caption(
    f"{NUM_EVALUATORS} evaluators · "
    f"{NUM_QUESTIONS} question(s) · "
    "1–5 ratings · 0 = not rated"
)


def atomic_save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(temp, path)


def load_ratings(path: Path) -> dict:
    if not path.exists():
        return {
            "schema_version": 1,
            "questions": DEFAULT_QUESTIONS[:],
            "ratings": {}
        }

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    questions = data.get("questions", [])

    data["questions"] = (
        questions[:NUM_QUESTIONS]
        + DEFAULT_QUESTIONS[len(questions):NUM_QUESTIONS]
    )

    return data


def find_run_folder(base: Path, run_name: str) -> Path:
    return base / run_name


def reference_path(run_dir: Path) -> Path | None:
    metadata = run_dir / "run.json"
    if not metadata.exists():
        return None
    try:
        with metadata.open("r", encoding="utf-8") as f:
            value = json.load(f).get("reference_audio")
        p = Path(value) if value else None
        return p if p and p.is_file() else None
    except (OSError, ValueError, TypeError):
        return None


def csv_bytes(data: dict) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)

    writer.writerow([
        "run",
        "candidate_step",
        "question_number",
        "question",
        "evaluator",
        "rating"
    ])

    for run, steps in RUNS.items():
        for step in steps:
            scores = (
                data.get("ratings", {})
                .get(run, {})
                .get(str(step), {})
            )

            for q_idx, question in enumerate(data["questions"]):
                for evaluator in range(1, NUM_EVALUATORS + 1):
                    rating = (
                        scores.get(str(q_idx), {})
                        .get(str(evaluator))
                    )

                    writer.writerow([
                        run,
                        step,
                        q_idx + 1,
                        question,
                        evaluator,
                        rating if rating is not None else ""
                    ])

    return stream.getvalue().encode("utf-8-sig")


with st.sidebar:
    st.header("Setup")
    base_text = st.text_input(
        "Exploration runs directory",
        value="output/exploration",
        help="Directory containing folders like 20261012_150555_jhin. Change to history/exploration if that is where yours are.",
    )
    base_dir = Path(base_text).expanduser()
    ratings_file = st.text_input("Ratings JSON", value="mos_results/ratings.json")
    save_path = Path(ratings_file).expanduser()
    run_name = st.selectbox("Voice / run", list(RUNS.keys()), format_func=lambda s: s.split("_", 2)[-1] + "  —  " + s)
    st.info("Use one shared copy of this app and ratings file. It is not designed for simultaneous editing from separate computers.")

try:
    data = load_ratings(save_path)
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"Could not load ratings: {exc}")
    st.stop()

st.subheader("Questions")
st.caption("Agree on the wording and scale before collecting ratings.")

with st.expander("Edit questions", expanded=False):
    with st.form("questions_form"):
        edited = [
            st.text_input(
                f"Question {i + 1}",
                value=data["questions"][i]
            )
            for i in range(NUM_QUESTIONS)
        ]

        if st.form_submit_button("Save questions"):
            if not all(x.strip() for x in edited):
                st.error("All questions need wording.")
            else:
                data["questions"] = [x.strip() for x in edited]
                atomic_save(save_path, data)
                st.success("Questions saved.")
                st.rerun()

run_dir = find_run_folder(base_dir, run_name)
steps = RUNS[run_name]
run_scores = data.setdefault("ratings", {}).setdefault(run_name, {})
completed = sum(
    all(run_scores.get(str(step), {}).get(str(q), {}).get(str(r)) in (1,2,3,4,5)
        for q in range(NUM_QUESTIONS) for r in range(1,NUM_EVALUATORS+1))
    for step in steps
)
st.progress(completed / len(steps), text=f"{completed} / {len(steps)} candidates fully rated by all three evaluators")

if not run_dir.is_dir():
    st.error(f"Run directory not found: {run_dir.resolve()}")
    st.info("Check the exploration directory in the sidebar. The run folders must be available locally.")

ref = reference_path(run_dir)
if ref is not None:
    with st.expander("Listen to reference voice"):
        st.audio(str(ref), format="audio/wav")

step = st.selectbox(
    "Candidate to evaluate",
    steps,
    format_func=lambda s: f"Candidate {s:03d}" + (" ✓" if all(
        run_scores.get(str(s), {}).get(str(q), {}).get(str(r)) in (1,2,3,4,5)
        for q in range(NUM_QUESTIONS) for r in range(1,NUM_EVALUATORS+1)) else ""),
)
path = run_dir / f"candidate_{step:03d}.wav"
if path.is_file():
    st.audio(str(path), format="audio/wav")
else:
    st.warning(f"Missing audio: {path}")

st.markdown("**Rating scale:** 1 = very poor · 2 = poor · 3 = fair · 4 = good · 5 = excellent. Leave 0 until rated.")
st.caption("Enter each person's score independently. The same three evaluator numbers should refer to the same people throughout.")


existing = run_scores.get(str(step), {})

with st.form(f"ratings_form_{run_name}_{step}"):
    submitted = {}

    for q_idx, question in enumerate(data["questions"]):
        st.markdown(f"**Q{q_idx + 1}. {question}**")

        columns = st.columns(NUM_EVALUATORS)
        submitted[str(q_idx)] = {}

        for r_idx, column in enumerate(columns, start=1):
            old = existing.get(str(q_idx), {}).get(str(r_idx))

            with column:
                submitted[str(q_idx)][str(r_idx)] = st.selectbox(
                    f"Evaluator {r_idx}",
                    [0, 1, 2, 3, 4, 5],
                    index=(
                        old
                        if isinstance(old, int) and 0 <= old <= 5
                        else 0
                    ),
                    key=f"{run_name}:{step}:{q_idx}:{r_idx}",
                )

    save_clicked = st.form_submit_button(
        "💾 Save ratings",
        type="primary"
    )

    if save_clicked:
        data["ratings"][run_name][str(step)] = {
            q: {
                r: (v if v > 0 else None)
                for r, v in scores.items()
            }
            for q, scores in submitted.items()
        }

        data["last_saved_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

        atomic_save(save_path, data)

        st.success(
            f"Saved ratings for candidate {step:03d} "
            f"to {save_path}"
        )

        st.rerun()

st.divider()
st.subheader("Export and summary")
all_rows = []

for r_name, candidate_steps in RUNS.items():
    for s in candidate_steps:
        entry = (
            data.get("ratings", {})
            .get(r_name, {})
            .get(str(s), {})
        )

        for q_idx, question in enumerate(data["questions"]):
            scores = [
                entry.get(str(q_idx), {}).get(str(i))
                for i in range(1, NUM_EVALUATORS + 1)
            ]

            valid = [
                v for v in scores
                if isinstance(v, int) and 1 <= v <= 5
            ]

            row = {
                "run": r_name,
                "candidate_step": s,
                "question": question,
            }

            for i, score in enumerate(scores, start=1):
                row[f"evaluator_{i}"] = score

            row["mean"] = (
                round(sum(valid) / len(valid), 3)
                if valid else None
            )

            row["ratings_count"] = len(valid)

            all_rows.append(row)

summary = pd.DataFrame(all_rows)
st.dataframe(summary[summary["run"] == run_name], use_container_width=True, hide_index=True)
st.download_button("⬇ Export all individual ratings (CSV)", csv_bytes(data),
                   file_name="dramatizer_mos_ratings.csv", mime="text/csv")
st.download_button("⬇ Export summary (CSV)", summary.to_csv(index=False).encode("utf-8-sig"),
                   file_name="dramatizer_mos_summary.csv", mime="text/csv")
st.caption("The CSV exports are snapshots. The JSON file is the persistent source of ratings.")

st.divider()
st.subheader("Finalize voice")

# Only use the currently selected run.
ratings_for_run = data.get("ratings", {}).get(run_name, {})

ranked = []

for candidate_step in RUNS[run_name]:
    entry = ratings_for_run.get(str(candidate_step), {})
    question_scores = entry.get("0", {})

    r1 = question_scores.get("1")
    r2 = question_scores.get("2")

    if (
        isinstance(r1, int) and 1 <= r1 <= 5
        and isinstance(r2, int) and 1 <= r2 <= 5
    ):
        ranked.append({
            "step": candidate_step,
            "evaluator_1": r1,
            "evaluator_2": r2,
            "mos": (r1 + r2) / 2,
        })

if len(ranked) != len(RUNS[run_name]):
    st.info(
        f"Waiting for ratings: {len(ranked)} / "
        f"{len(RUNS[run_name])} candidates complete."
    )

else:
    metadata_path = run_dir / "run.json"

    if not metadata_path.is_file():
        st.error(f"Missing run metadata: {metadata_path}")

    else:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        candidates_by_step = {
            int(c["step"]): c
            for c in metadata["candidates"]
        }

        eligible = []

        for item in ranked:
            candidate = candidates_by_step.get(item["step"])

            if candidate is None:
                st.error(
                    f"Candidate {item['step']} missing from run.json."
                )
                break

            if candidate.get("status") != "accepted":
                st.error(
                    f"Candidate {item['step']} is not accepted."
                )
                break

            score = candidate.get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                st.error(
                    f"Candidate {item['step']} has no valid reward score."
                )
                break

            eligible.append({
                **item,
                "automatic_score": score,
                "metadata": candidate,
            })

        if len(eligible) == len(ranked):
            eligible.sort(
                key=lambda x: (
                    -x["mos"],
                    -x["automatic_score"],
                    x["step"]
                )
            )

            winner = eligible[0]
            winning_metadata = winner["metadata"]

            st.success("All candidates have been evaluated!")

            st.dataframe(
                pd.DataFrame([
                    {
                        "Candidate": x["step"],
                        "Evaluator 1": x["evaluator_1"],
                        "Evaluator 2": x["evaluator_2"],
                        "MOS": x["mos"],
                        "Automatic score": x["automatic_score"],
                    }
                    for x in eligible
                ]),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown(
                f"**Winner: Candidate {winner['step']:03d}** "
                f"— MOS {winner['mos']:.2f}/5"
            )

            st.json(winning_metadata["genome"])

            history_path = Path("history/voices.json")
            voice_id = metadata["voice_id"]
            reference_audio = Path(metadata["reference_audio"])

            # Check whether this voice already has a stored winner.
            # Avoid importing/loading GenomeHistory until the user
            # actually chooses to save.
            existing_records = []
            if history_path.exists():
                with history_path.open("r", encoding="utf-8") as f:
                    existing_records = json.load(f)

            already_saved = any(
                record.get("voice_id") == voice_id
                for record in existing_records
            )

            if already_saved:
                st.warning(
                    f"History already contains voice {voice_id!r}. "
                    "Saving another winner is disabled."
                )

            elif not reference_audio.is_file():
                st.error(
                    f"Reference audio not found: {reference_audio}. "
                    "Check the path recorded in run.json."
                )

            elif st.button(
                "Confirm and save winner to history",
                type="primary"
            ):
                # Keep a backup of the previous history file.
                if history_path.exists():
                    import shutil
                    shutil.copy2(
                        history_path,
                        history_path.with_suffix(".json.bak")
                    )

                history = GenomeHistory(str(history_path))

                history.store_winner(
                    voice_id=voice_id,
                    reference_audio=reference_audio,
                    candidate=Candidate(winning_metadata["genome"]),
                    mos=winner["mos"],
                    wer=winning_metadata["wer"],
                    speaker_similarity_score=(
                        winning_metadata["speaker_similarity"]
                    ),
                )

                st.success(
                    f"Saved candidate {winner['step']:03d} "
                    f"for {voice_id} to {history_path}"
                )
                st.rerun()
