from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Union

logger = logging.getLogger(__name__)


# CosyVoice 2 point-event tokens (see CosyVoice2Tokenizer in
# cosyvoice/tokenizer/tokenizer.py), keyed by feature name. Each key is also
# registered in the Option Table below.
EVENT_TAGS = {
    "breath": "[breath]",
    "quick_breath": "[quick_breath]",
    "noise": "[noise]",
    "laughter": "[laughter]",
    "cough": "[cough]",
    "clucking": "[clucking]",  # tongue click
    "accent": "[accent]",
    "hissing": "[hissing]",
    "sigh": "[sigh]",
    "vocalized_noise": "[vocalized-noise]",
    "lipsmack": "[lipsmack]",
    "mn": "[mn]",  # hesitation hum
}

# Alternate spellings accepted by add_micro_expression
EVENT_ALIASES = {
    "laugh": "laughter",
    "vocalized-noise": "vocalized_noise",
}

SPAN_TAGS = ("strong", "laughter")

# Soft pauses are punctuation only: a hint to the model, length not controllable
SOFT_PAUSES = {
    "dramatic": "...",
    "realization": " —",  # Em-dash for sudden stop
}

_PUNCT = ".!?,"

_EVENT = "|".join(re.escape(tag) for tag in EVENT_TAGS.values())
_SPAN = "|".join(SPAN_TAGS)
_PAUSE = r"⟦pause:(\d+)⟧"

PAUSE_MARKER_RE = re.compile(_PAUSE)
_SPAN_TAG_RE = re.compile(rf"<(/?)({_SPAN})>")
_MARKUP_RE = re.compile(rf" ?(?:{_EVENT})|</?(?:{_SPAN})>|{_PAUSE}")
_OPENERS_BEFORE_RE = re.compile(rf"(?:<(?:{_SPAN})>)*$")
_CLOSERS_RE = re.compile(rf"(?:</(?:{_SPAN})>)*")
_CLOSERS_AND_EVENTS_RE = re.compile(rf"(?: ?(?:{_EVENT})|</(?:{_SPAN})>)*")


class TargetNotFoundError(ValueError):
    """Raised when an operation's target is not present in the text."""


@dataclass
class TTSPlan:
    text: str                 # markup text, may contain pause markers
    speed: float = 1.0        # passed to inference as speed=
    instruct: str | None = None   # optional natural-language instruction
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TextSegment:
    text: str


@dataclass(frozen=True)
class SilenceSegment:
    ms: int


Segment = Union[TextSegment, SilenceSegment]


def _word_pattern(target_word: str) -> str:
    """Builds a whole-word pattern for the target.

    A plain \\b fails when the target starts or ends with punctuation
    (e.g. "off."), so each edge only gets a boundary check if it is a word
    character.
    """
    prefix = r'(?<!\w)' if re.match(r'\w', target_word) else ''
    suffix = r'(?!\w)' if re.search(r'\w$', target_word) else ''
    return rf'{prefix}({re.escape(target_word)}){suffix}'


def _strip_markup(text: str) -> tuple[str, list[int]]:
    """Returns the text without tags or pause markers, and each kept character's original index."""
    index = []
    pos = 0
    for m in _MARKUP_RE.finditer(text):
        index.extend(range(pos, m.start()))
        pos = m.end()
    index.extend(range(pos, len(text)))
    return ''.join(text[i] for i in index), index


def _locate(text: str, target: str, trailing_punct: bool = False) -> tuple[int, int, int | None]:
    """Finds the first match of target in the tag-stripped text.

    Returns (start, end, punct) as offsets into the original text, where punct
    is the offset of the match's trailing punctuation, or None if it has none.
    With trailing_punct, punctuation following the target is part of the match.
    """
    if not target:
        raise ValueError("target must not be empty")
    plain, index = _strip_markup(text)
    pattern = _word_pattern(target) + (rf'[{_PUNCT}]*' if trailing_punct else '')
    m = re.search(pattern, plain)
    if not m:
        raise TargetNotFoundError(f"target {target!r} not found")
    punct = re.search(rf'[{_PUNCT}]+$', m.group(0))
    punct_pos = index[m.start() + punct.start()] if punct else None
    return index[m.start()], index[m.end() - 1] + 1, punct_pos


def _insert_tag(text: str, target_word: str, tag: str) -> str:
    """Inserts a point-event tag after the first match of target_word, before any trailing punctuation."""
    _, end, punct = _locate(text, target_word)
    pos = punct if punct is not None else _CLOSERS_AND_EVENTS_RE.match(text, end).end()
    return f'{text[:pos]} {tag}{text[pos:]}'


def _balanced(fragment: str) -> bool:
    """True if every span tag in the fragment is opened and closed inside it."""
    stack = []
    for closing, name in _SPAN_TAG_RE.findall(fragment):
        if not closing:
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
    return not stack


def _wrap(text: str, target: str, name: str) -> str:
    """Wraps the first match of target in a span tag, keeping existing spans properly nested."""
    start, end, _ = _locate(text, target)
    left = _OPENERS_BEFORE_RE.search(text, 0, start).start()
    right = _CLOSERS_RE.match(text, end).end()
    for s, e in ((start, end), (left, end), (start, right), (left, right)):
        if _balanced(text[s:e]):
            return f'{text[:s]}<{name}>{text[s:e]}</{name}>{text[e:]}'
    raise ValueError(f"target {target!r} partially overlaps an existing span")


def add_micro_expression(text: str, target_word: str, expr_type: str = "laughter") -> str:
    """Injects a CosyVoice event tag (e.g. [laughter], [breath]) after a target word."""
    key = expr_type.lower()
    key = EVENT_ALIASES.get(key, key)
    if key not in EVENT_TAGS:
        valid = ", ".join(sorted([*EVENT_TAGS, *EVENT_ALIASES]))
        raise ValueError(f"unknown expr_type {expr_type!r}; valid keys: {valid}")

    return _insert_tag(text, target_word, EVENT_TAGS[key])


def _event_feature(tag: str):
    """Builds the Option Table function for a single event tag."""
    def add_event(text: str, target_word: str) -> str:
        return _insert_tag(text, target_word, tag)
    add_event.__doc__ = f"Injects {tag} after a target word."
    return add_event


def add_laughing_speech(text: str, target_phrase: str) -> str:
    """Wraps a phrase in <laughter> tags so it is spoken while laughing."""
    return _wrap(text, target_phrase, "laughter")


def add_emphasis(text: str, target_word: str) -> str:
    """Wraps a specific word in <strong> tags."""
    return _wrap(text, target_word, "strong")


def add_pause(text: str, target_word: str, pause_type: str = "dramatic", pause_ms: int | None = None) -> str:
    """Injects a pause after a target word.

    pause_ms gives a timed pause: a private ⟦pause:N⟧ marker that
    split_on_pauses turns into silence. Otherwise pause_type picks a soft
    pause: "dramatic" or "realization" punctuation, or an audible "breath".
    """
    if pause_ms is not None:
        if isinstance(pause_ms, bool) or not isinstance(pause_ms, int) or pause_ms < 0:
            raise ValueError(f"pause_ms must be a non-negative int, got {pause_ms!r}")
        _, end, _ = _locate(text, target_word, trailing_punct=True)
        pos = _CLOSERS_RE.match(text, end).end()
        return f'{text[:pos]}⟦pause:{pause_ms}⟧{text[pos:]}'  #We have to manually insert the pause here, CozyVoice 2 does not support automatic pause insertion.

    if pause_type == "breath":
        return _insert_tag(text, target_word, EVENT_TAGS["breath"])
    if pause_type not in SOFT_PAUSES:
        valid = ", ".join(sorted([*SOFT_PAUSES, "breath"]))
        raise ValueError(f"unknown pause_type {pause_type!r}; valid keys: {valid} (or pass pause_ms)")

    _, end, punct = _locate(text, target_word, trailing_punct=True)
    if punct is None:
        punct = end = _CLOSERS_RE.match(text, end).end()
    return f'{text[:punct]}{SOFT_PAUSES[pause_type]}{text[end:]}'


def speed_instruction(rate: float) -> str | None:
    """Maps a rate to a natural-language instruction for inference_instruct2."""
    if rate >= 1.5:
        return "Speaking very fast"
    if rate > 1.0:
        return "Speaking fast"
    if rate == 1.0:
        return None
    if rate > 0.7:
        return "Speaking slowly"
    return "Speaking very slowly"


def adjust_speed(plan: TTSPlan, rate: float = 1.0, mode: str = "param") -> None:
    """Sets the speaking rate on the plan; the text is left untouched.

    "param" stores the rate for the inference call's speed argument;
    "instruct" stores a natural-language instruction instead. Never both.
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError(f"rate must be a positive number, got {rate!r}")
    if mode == "param":
        plan.speed, plan.instruct = float(rate), None
    elif mode == "instruct":
        plan.speed, plan.instruct = 1.0, speed_instruction(rate)
    else:
        raise ValueError(f"unknown speed mode {mode!r}; valid keys: instruct, param")


# The Option Table
FORMATTING_INDEX = {
    "micro_expression": add_micro_expression,
    "emphasis": add_emphasis,
    "pause": add_pause,
    "laughing_speech": add_laughing_speech,
    **{feature: _event_feature(tag) for feature, tag in EVENT_TAGS.items()},
}

# Features that set data on the plan instead of rewriting the text
PLAN_INDEX = {
    "speed": adjust_speed,
}


def process_tts_string(base_text: str, operations: list[dict], strict: bool = False) -> TTSPlan:
    """
    Applies a list of formatting operations to the base text.

    Expected operation format:
    {"feature": "feature_name", "kwargs": {"param1": "value1"}}

    Unknown features and failed operations are skipped and recorded in
    plan.warnings, or raised in strict mode.
    """
    plan = TTSPlan(text=base_text)

    for op in operations:
        feature = None
        try:
            feature = op.get("feature")
            kwargs = op.get("kwargs") or {}

            # Lookup the function in the index
            if feature in PLAN_INDEX:
                PLAN_INDEX[feature](plan, **kwargs)
            elif feature in FORMATTING_INDEX:
                plan.text = FORMATTING_INDEX[feature](plan.text, **kwargs)
            else:
                raise ValueError("feature not found in Option Table")
        except Exception as e:
            if strict:
                raise
            message = f"{feature}: {e}"
            logger.warning(message)
            plan.warnings.append(message)

    return plan


def split_on_pauses(text: str) -> list[Segment]:
    """Splits processed text on timed-pause markers into text and silence segments.

    A marker inside an open <strong> or <laughter> span closes the span before
    the marker and reopens it after. Empty text segments are dropped.
    """
    segments: list[Segment] = []
    open_spans: list[str] = []

    # re.split with one capture group alternates text, ms, text, ms, ...
    for i, piece in enumerate(PAUSE_MARKER_RE.split(text)):
        if i % 2:
            ms = int(piece)
            if segments and isinstance(segments[-1], SilenceSegment):
                segments[-1] = SilenceSegment(segments[-1].ms + ms)
            elif ms:
                segments.append(SilenceSegment(ms))
            continue

        reopened = ''.join(f'<{name}>' for name in open_spans)
        for closing, name in _SPAN_TAG_RE.findall(piece):
            if not closing:
                open_spans.append(name)
            elif open_spans and open_spans[-1] == name:
                open_spans.pop()
        closed = ''.join(f'</{name}>' for name in reversed(open_spans))

        if _SPAN_TAG_RE.sub('', piece).strip():
            segments.append(TextSegment(f'{reopened}{piece.strip()}{closed}'))

    return segments


def synthesize(cosyvoice, plan: TTSPlan, prompt_wav, prompt_text: str | None = None, **kwargs):
    """Synthesizes a plan with a loaded CosyVoice2 model, stitching audio and silence.

    Uses inference_instruct2 when the plan has an instruction, inference_zero_shot
    when prompt_text is given, and inference_cross_lingual otherwise. Extra
    kwargs (e.g. text_frontend=False) are passed to the inference call.
    """
    if kwargs.pop("stream", False) and plan.speed != 1.0:
        raise ValueError(f"speed={plan.speed} requires stream=False; CosyVoice 2 cannot change speed while streaming")

    # Imported here so the text logic stays importable without torch
    import torch

    kwargs.update(stream=False, speed=plan.speed)
    chunks = []
    for segment in split_on_pauses(plan.text):
        if isinstance(segment, SilenceSegment):
            chunks.append(torch.zeros(1, int(cosyvoice.sample_rate * segment.ms / 1000)))
        elif plan.instruct:
            outputs = cosyvoice.inference_instruct2(segment.text, plan.instruct, prompt_wav, **kwargs)
            chunks.extend(out["tts_speech"] for out in outputs)
        elif prompt_text:
            outputs = cosyvoice.inference_zero_shot(segment.text, prompt_text, prompt_wav, **kwargs)
            chunks.extend(out["tts_speech"] for out in outputs)
        else:
            outputs = cosyvoice.inference_cross_lingual(segment.text, prompt_wav, **kwargs)
            chunks.extend(out["tts_speech"] for out in outputs)

    return torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. The raw input string
    raw_text = "I can't believe we actually pulled that off. It was incredible."

    # 2. The requested formatting operations (e.g., generated by a UI or LLM)
    requested_operations = [
        {"feature": "emphasis", "kwargs": {"target_word": "actually"}},
        {"feature": "pause", "kwargs": {"target_word": "off.", "pause_type": "dramatic"}},
        {"feature": "micro_expression", "kwargs": {"target_word": "incredible.", "expr_type": "laughter"}},
        {"feature": "speed", "kwargs": {"rate": 1.1}},
    ]

    # 3. Process the string
    print(process_tts_string(raw_text, requested_operations))
