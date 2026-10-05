# AHJN-and-MING

`tts_formatter.py` is an option-table pipeline that injects expressive markup into plain text before it is sent to CosyVoice 2. Operations arrive as a list of `{"feature": ..., "kwargs": {...}}` dicts and are applied in order. It needs neither torch nor CosyVoice. `synthesize`, in `tts_synthesis.py`, is the only part that needs a loaded model.

## Usage

```python
from tts_formatter import process_tts_string
from tts_synthesis import synthesize

plan = process_tts_string(
    "I can't believe we actually pulled that off. It was incredible.",
    [
        {"feature": "emphasis", "kwargs": {"target": "actually"}},
        {"feature": "pause", "kwargs": {"target": "off", "pause_type": "dramatic"}},
        {"feature": "laughter", "kwargs": {"target": "incredible"}},
        {"feature": "speed", "kwargs": {"rate": 1.1}},
    ],
)
# plan.text  -> "I can't believe we <strong>actually</strong> pulled that off... It was incredible [laughter]."
# plan.speed -> 1.1

audio = synthesize(cosyvoice, plan, "reference.wav")   # cosyvoice is a loaded CosyVoice2 model
```

Running the module directly (`python tts_formatter.py`) prints the plan for this demo input.

## The plan

`process_tts_string(base_text, operations, strict=False)` returns a `TTSPlan`:

| Field | Meaning |
|---|---|
| `text` | Markup text; may contain timed-pause markers |
| `speed` | Passed to inference as `speed=` (default `1.0`) |
| `instruct` | Optional natural-language instruction, or `None` |
| `warnings` | Everything that was skipped, clamped, removed or is experimental |

## Features

| Feature | Kwargs | Effect |
|---|---|---|
| `emphasis` | `target` | Wraps the target in `<strong>…</strong>` |
| `laughing_speech` | `target` | Wraps the target in `<laughter>…</laughter>` |
| `pause` | `target`, `pause_type` or `pause_ms` | Soft or timed pause after the target |
| `breath`, `quick_breath`, `noise`, `laughter`, `cough`, `clucking`, `accent`, `hissing`, `sigh`, `vocalized_noise`, `lipsmack`, `mn` | `target` | Inserts the matching event token (`[breath]`, `[vocalized-noise]`, …) after the target |
| `speed` | `rate`, `mode` | Sets `plan.speed`, or a speed instruction |
| `instruct` | `text` | Sets `plan.instruct` |

`micro_expression` (`target_word`, `expr_type`) still works for older callers; `expr_type` also accepts the aliases `laugh` and `vocalized-noise`. An unknown `expr_type` raises `ValueError` listing the valid keys.

`describe_options()` returns this table as a list of dicts (`feature`, `description`, `kwargs`, `experimental`), built from the function signatures, for use in an LLM prompt or a UI. Each kwarg carries its JSON type and, where they apply, `required`, `default`, `enum`, `minimum` and `maximum`. `micro_expression` is left out.

### Targets

- Every targeted feature accepts `target`. The older `target_word` and `target_phrase` still work; passing both `target` and the older name is an error.
- Matching is whole-word, on the text with tags and pause markers stripped, so a target still matches after an earlier operation has added markup inside or next to it. `off` does not match inside `offer`.
- `occurrence` (1-based positive int, default 1) picks a repeated word. A missing target, or asking for more occurrences than exist, raises `TargetNotFoundError`.
- A span that would partially overlap an existing span is an error; spans are always properly nested.

### Event tokens

- The token goes after the target and before any trailing punctuation: `incredible [laughter].`
- Only `laughter` and `breath` are in `VERIFIED_EVENTS`. Every other token still works, but adds one "experimental" warning per token to `plan.warnings`.

### Pauses

- **Soft pause** (`pause_type`, default `"dramatic"`): a hint to the model; its length is not controllable.
  - `"dramatic"` gives `...`, `"realization"` gives ` —`, `"breath"` inserts `[breath]`.
  - Trailing `.` or `,` is replaced by the marker: `off.` becomes `off...`.
  - `?` and `!` are kept: dramatic gives `off...?`. A realization pause on `?` or `!` is an error.
- **Timed pause** (`pause_ms`): inserts the private marker `⟦pause:N⟧` after the target and its punctuation. The marker never reaches the model; `split_on_pauses` turns it into silence.
  - A warning is added when the pause is not at a sentence end, or when a text segment next to it has fewer than three words.

### Speed and instructions

- `speed` with the default `"mode": "param"` stores `rate` in `plan.speed` (`1.0` is normal, `2.0` is twice as fast).
- `speed` with `"mode": "instruct"` resets `plan.speed` to `1.0` and stores a phrase in `plan.instruct` instead:

  | `rate` | Phrase |
  |---|---|
  | ≥ 1.5 | `Speaking very fast` |
  | > 1.0 | `Speaking fast` |
  | 1.0 | none |
  | > 0.7 | `Speaking slowly` |
  | ≤ 0.7 | `Speaking very slowly` |

  A speed phrase and a numeric speed are never both applied.
- `instruct` sets a free-text instruction such as `"Speak sadly"`: non-empty, at most 200 characters, no `<|` or `|>`. It may be combined with a numeric speed.
- A later instruction replaces an earlier one, with a warning.

### Bounds

`rate` is limited to 0.5–2.0 and `pause_ms` to 0–5000 (`BOUNDS`). Out-of-range values are clamped with a warning, or raise in strict mode. Wrong types always raise.

## Input text

`base_text` must be plain text. `sanitize_text(text)` returns the cleaned text and the list of removed pieces:

- Event tokens, `<strong>` / `<laughter>` tags, pause markers and `<|…|>` are removed.
- Any other angle-bracket tag (opening, closing or self-closing, such as `<speed rate='2'>` or `<break time='1s'/>`) is removed too. The text between tags is kept and a double space left behind is collapsed.
- A plain `<` or `>`, as in `3 < 5`, and other square-bracketed text, such as `[citation needed]`, are left alone.

`process_tts_string` records the removal as a `base_text: ` warning, or raises in strict mode.

## Errors and warnings

- `strict=False` (default): unknown features and failed operations are skipped and recorded in `plan.warnings`.
- `strict=True`: they raise. Advisories (experimental tokens, timed-pause placement) never raise.
- Warnings about an operation start with `op N (feature): `, counting from 1. Others start with `base_text: ` or `synthesize: `, or are advisories about the finished text.
- Every warning is also sent to `logging`.
