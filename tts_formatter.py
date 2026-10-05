from __future__ import annotations

import inspect
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

# Events confirmed by listening; the rest still work but add a warning to the plan
VERIFIED_EVENTS = {"laughter", "breath"}

SPAN_TAGS = ("strong", "laughter")

# Soft pauses are punctuation only: a hint to the model, length not controllable
SOFT_PAUSES = {
    "dramatic": "...",
    "realization": " —",  # Em-dash for sudden stop
}

# Inclusive ranges for numeric kwargs; process_tts_string clamps to these
BOUNDS = {
    "rate": (0.5, 2.0),
    "pause_ms": (0, 5000),
}

END_OF_PROMPT = "<|endofprompt|>"
INSTRUCT_MAX_CHARS = 200
SPEED_INSTRUCTIONS = ("Speaking very fast", "Speaking fast", "Speaking slowly", "Speaking very slowly")

# With text_frontend=False the model gets each text segment unsplit; the
# frontend would have cut it into pieces of roughly this size
FRONTEND_OFF_MAX_WORDS = 80

_PUNCT = ".!?,"

_EVENT = "|".join(re.escape(tag) for tag in EVENT_TAGS.values())
_SPAN = "|".join(SPAN_TAGS)
_PAUSE = r"⟦pause:(\d+)⟧"
# Any other SSML/HTML-style tag: opening, closing or self-closing. A bare "<"
# or ">" (as in "3 < 5") is not followed by a tag name and does not match.
_OTHER_TAG = r"</?[A-Za-z][^<>]*>"
# Han and kana are written without spaces, so they take no word boundary
_CJK = r"぀-ヿ㐀-䶿一-鿿豈-﫿"
_SPACED_WORD_CHAR = rf"[^\W{_CJK}]"

PAUSE_MARKER_RE = re.compile(_PAUSE)
_EVENT_RE = re.compile(_EVENT)
_CJK_RE = re.compile(rf"[{_CJK}]")
# Same test as contains_chinese in the CosyVoice frontend, which picks its Chinese path
_CHINESE_RE = re.compile(r"[一-鿿]")
_RESERVED_RE = re.compile(rf" ?(?:{_EVENT})|</?(?:{_SPAN})>|{_PAUSE}|<\|[^|]*\|>|<\||\|>|{_OTHER_TAG}")
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
    character. CJK characters are exempt on both sides of the check: they are
    written without spaces, so a target may sit anywhere in a CJK sentence.
    """
    prefix = rf'(?<!{_SPACED_WORD_CHAR})' if re.match(_SPACED_WORD_CHAR, target_word) else ''
    suffix = rf'(?!{_SPACED_WORD_CHAR})' if re.search(rf'{_SPACED_WORD_CHAR}$', target_word) else ''
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


def _locate(text: str, target: str, trailing_punct: bool = False, occurrence: int = 1) -> tuple[int, int, int | None]:
    """Finds the occurrence-th match (1-based) of target in the tag-stripped text.

    Returns (start, end, punct) as offsets into the original text, where punct
    is the offset of the match's trailing punctuation, or None if it has none.
    With trailing_punct, punctuation following the target is part of the match.
    """
    if not target:
        raise ValueError("target must not be empty")
    if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 1:
        raise ValueError(f"occurrence must be a positive int, got {occurrence!r}")
    plain, index = _strip_markup(text)
    pattern = _word_pattern(target) + (rf'[{_PUNCT}]*' if trailing_punct else '')
    matches = list(re.finditer(pattern, plain))
    if not matches:
        raise TargetNotFoundError(f"target {target!r} not found")
    if len(matches) < occurrence:
        raise TargetNotFoundError(f"only {len(matches)} occurrences of {target!r}, asked for occurrence {occurrence}")
    m = matches[occurrence - 1]
    punct = re.search(rf'[{_PUNCT}]+$', m.group(0))
    punct_pos = index[m.start() + punct.start()] if punct else None
    return index[m.start()], index[m.end() - 1] + 1, punct_pos


def _insert_tag(text: str, target_word: str, tag: str, occurrence: int = 1) -> str:
    """Inserts a point-event tag after a match of target_word, before any trailing punctuation."""
    _, end, punct = _locate(text, target_word, occurrence=occurrence)
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


def _wrap(text: str, target: str, name: str, occurrence: int = 1) -> str:
    """Wraps a match of target in a span tag, keeping existing spans properly nested."""
    start, end, _ = _locate(text, target, occurrence=occurrence)
    left = _OPENERS_BEFORE_RE.search(text, 0, start).start()
    right = _CLOSERS_RE.match(text, end).end()
    for s, e in ((start, end), (left, end), (start, right), (left, right)):
        if _balanced(text[s:e]):
            return f'{text[:s]}<{name}>{text[s:e]}</{name}>{text[e:]}'
    raise ValueError(f"target {target!r} partially overlaps an existing span")


def add_micro_expression(text: str, target_word: str, expr_type: str = "laughter", occurrence: int = 1) -> str:
    """Injects a CosyVoice event tag (e.g. [laughter], [breath]) after a target word."""
    key = expr_type.lower()
    key = EVENT_ALIASES.get(key, key)
    if key not in EVENT_TAGS:
        valid = ", ".join(sorted([*EVENT_TAGS, *EVENT_ALIASES]))
        raise ValueError(f"unknown expr_type {expr_type!r}; valid keys: {valid}")

    return _insert_tag(text, target_word, EVENT_TAGS[key], occurrence)


def _event_feature(tag: str):
    """Builds the Option Table function for a single event tag."""
    def add_event(text: str, target_word: str, occurrence: int = 1) -> str:
        return _insert_tag(text, target_word, tag, occurrence)
    add_event.__doc__ = f"Injects {tag} after a target word."
    return add_event


def add_laughing_speech(text: str, target_phrase: str, occurrence: int = 1) -> str:
    """Wraps a phrase in <laughter> tags so it is spoken while laughing."""
    return _wrap(text, target_phrase, "laughter", occurrence)


def add_emphasis(text: str, target_word: str, occurrence: int = 1) -> str:
    """Wraps a specific word or phrase in <strong> tags."""
    return _wrap(text, target_word, "strong", occurrence)


def add_pause(text: str, target_word: str, pause_type: str = "dramatic", pause_ms: int | None = None,
              occurrence: int = 1) -> str:
    """Injects a pause after a target word.

    pause_ms gives a timed pause: a private ⟦pause:N⟧ marker that
    split_on_pauses turns into silence. Otherwise pause_type picks a soft
    pause: "dramatic" or "realization" punctuation, or an audible "breath".
    """
    if pause_ms is not None:
        if isinstance(pause_ms, bool) or not isinstance(pause_ms, int) or pause_ms < 0:
            raise ValueError(f"pause_ms must be a non-negative int, got {pause_ms!r}")
        _, end, _ = _locate(text, target_word, trailing_punct=True, occurrence=occurrence)
        pos = _CLOSERS_RE.match(text, end).end()
        return f'{text[:pos]}⟦pause:{pause_ms}⟧{text[pos:]}'  #We have to manually insert the pause here, CozyVoice 2 does not support automatic pause insertion.

    if pause_type == "breath":
        return _insert_tag(text, target_word, EVENT_TAGS["breath"], occurrence)
    if pause_type not in SOFT_PAUSES:
        valid = ", ".join(sorted([*SOFT_PAUSES, "breath"]))
        raise ValueError(f"unknown pause_type {pause_type!r}; valid keys: {valid} (or pass pause_ms)")
    # The frontend's Chinese path rewrites " - " to a comma and "——" to a space
    if pause_type == "realization" and _CHINESE_RE.search(text):
        raise ValueError("a realization pause is not supported in text containing Chinese; "
                         "use pause_type 'breath' or pause_ms")

    marker = SOFT_PAUSES[pause_type]
    _, end, punct = _locate(text, target_word, trailing_punct=True, occurrence=occurrence)
    if punct is None:
        pos = _CLOSERS_RE.match(text, end).end()
        return f'{text[:pos]}{marker}{text[pos:]}'
    # ? and ! carry the intonation cue, so they are kept rather than replaced
    if re.search(r'[?!]', text[punct:end]):
        if pause_type == "realization":
            raise ValueError(f"a realization pause cannot follow {text[punct:end]!r}; use dramatic or breath")
        return f'{text[:punct]}{marker}{text[punct:]}'
    return f'{text[:punct]}{marker}{text[end:]}'


def speed_instruction(rate: float) -> str | None:
    """Maps a rate to a natural-language instruction for inference_instruct2."""
    very_fast, fast, slowly, very_slowly = SPEED_INSTRUCTIONS
    if rate >= 1.5:
        return very_fast
    if rate > 1.0:
        return fast
    if rate == 1.0:
        return None
    if rate > 0.7:
        return slowly
    return very_slowly


def adjust_speed(plan: TTSPlan, rate: float = 1.0, mode: str = "param") -> None:
    """Sets the speaking rate; 1.0 is normal, 2.0 is twice as fast.

    "param" stores the rate for the inference call's speed argument;
    "instruct" stores a natural-language speed instruction instead and resets
    the numeric speed. A speed instruction and a numeric speed are never both
    applied; an instruction set by the "instruct" feature is left alone.
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError(f"rate must be a positive number, got {rate!r}")
    if mode == "param":
        plan.speed = float(rate)
        if plan.instruct in SPEED_INSTRUCTIONS:
            plan.instruct = None
    elif mode == "instruct":
        plan.speed = 1.0
        instruction = speed_instruction(rate)
        if instruction or plan.instruct in SPEED_INSTRUCTIONS:
            plan.instruct = instruction
    else:
        raise ValueError(f"unknown speed mode {mode!r}; valid keys: instruct, param")


def set_instruct(plan: TTSPlan, text: str) -> None:
    """Sets a natural-language instruction for the whole utterance, e.g. "Speak sadly"."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"instruct text must be a non-empty string, got {text!r}")
    if "<|" in text or "|>" in text:
        raise ValueError("instruct text must not contain <| or |>")
    if len(text) > INSTRUCT_MAX_CHARS:
        raise ValueError(f"instruct text is {len(text)} characters; the limit is {INSTRUCT_MAX_CHARS}")
    plan.instruct = text.strip()


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
    "instruct": set_instruct,
}

# Kept working for older callers, but left out of describe_options
_HIDDEN_FEATURES = {"micro_expression"}

_TARGET_PARAMS = ("target_word", "target_phrase")
_KWARG_ENUMS = {
    "pause_type": sorted([*SOFT_PAUSES, "breath"]),
    "mode": ["param", "instruct"],
}
_JSON_TYPES = {"str": "string", "int": "integer", "float": "number"}


def _target_param(func) -> str | None:
    """Returns the name of the function's target parameter, if it has one."""
    return next((name for name in inspect.signature(func).parameters if name in _TARGET_PARAMS), None)


def describe_options() -> list[dict]:
    """Describes the Option Table for an LLM or UI, built from the functions themselves.

    Every target parameter is advertised as "target". Experimental features
    are event tokens that have not yet been verified by listening.
    """
    options = []
    for feature, func in {**FORMATTING_INDEX, **PLAN_INDEX}.items():
        if feature in _HIDDEN_FEATURES:
            continue
        kwargs = {}
        # The first parameter is the text or the plan, supplied by process_tts_string
        for param in list(inspect.signature(func).parameters.values())[1:]:
            spec = {"type": _JSON_TYPES[str(param.annotation).split(" | ")[0]]}
            if param.default is inspect.Parameter.empty:
                spec["required"] = True
            elif param.default is not None:
                spec["default"] = param.default
            if param.name in _KWARG_ENUMS:
                spec["enum"] = _KWARG_ENUMS[param.name]
            if param.name in BOUNDS:
                spec["minimum"], spec["maximum"] = BOUNDS[param.name]
            kwargs["target" if param.name in _TARGET_PARAMS else param.name] = spec
        options.append({
            "feature": feature,
            "description": inspect.getdoc(func).split("\n\n")[0].replace("\n", " "),
            "kwargs": kwargs,
            "experimental": feature in EVENT_TAGS and feature not in VERIFIED_EVENTS,
        })
    return options


def sanitize_text(text: str) -> tuple[str, list[str]]:
    """Removes model tokens, pause markers, <|…|> and any angle-bracket tag from raw input text.

    Returns the cleaned text and the pieces that were removed. The text between
    tags is kept, and a double space left behind by a removal is collapsed.
    Other square-bracketed text and a plain "<" or ">" are left alone.
    """
    clean, removed, pos = '', [], 0
    for m in _RESERVED_RE.finditer(text):
        clean += text[pos:m.start()]
        removed.append(m.group(0).strip())
        pos = m.end()
        if (not clean or clean.endswith(' ')) and text.startswith(' ', pos):
            pos += 1
    return clean + text[pos:], removed


def _clamp_kwargs(kwargs: dict, strict: bool) -> list[str]:
    """Clamps numeric kwargs to BOUNDS in place and returns a note per change; raises in strict mode."""
    notes = []
    for key, (low, high) in BOUNDS.items():
        value = kwargs.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        clamped = min(max(value, low), high)
        if clamped != value:
            if strict:
                raise ValueError(f"{key}={value!r} is outside {low} to {high}")
            kwargs[key] = clamped
            notes.append(f"{key}={value!r} is outside {low} to {high}; clamped to {clamped!r}")
    return notes


def _advisories(text: str) -> list[str]:
    """Notes on the finished text that never fail an operation: unverified tokens and risky timed pauses."""
    notes = []
    verified = {EVENT_TAGS[key] for key in VERIFIED_EVENTS}
    for tag in dict.fromkeys(_EVENT_RE.findall(text)):
        if tag not in verified:
            notes.append(f"{tag} is experimental: not yet verified by listening")

    pieces = PAUSE_MARKER_RE.split(text)
    if len(pieces) == 1:
        return notes
    for before in pieces[:-1:2]:
        plain = _strip_markup(before)[0].rstrip()
        if plain and plain[-1] not in ".?!":
            notes.append(f"timed pause after {plain.split()[-1]!r} is not at a sentence end; "
                         "intonation may reset there, consider a soft pause")
    for segment in split_on_pauses(text):
        if isinstance(segment, TextSegment) and len(_strip_markup(segment.text)[0].split()) < 3:
            notes.append(f"text segment {segment.text!r} next to a timed pause has fewer than three words; "
                         "short segments may sound poor, consider a soft pause")
    return notes


def _warn(plan: TTSPlan, message: str) -> None:
    logger.warning(message)
    plan.warnings.append(message)


def process_tts_string(base_text: str, operations: list[dict], strict: bool = False) -> TTSPlan:
    """
    Applies a list of formatting operations to the base text.

    Expected operation format:
    {"feature": "feature_name", "kwargs": {"param1": "value1"}}

    Every targeted feature also accepts "target" in place of its own target
    kwarg, and "occurrence" (1-based) to pick a repeated word.

    Markup already present in base_text, unknown features, failed operations
    and out-of-range numbers are fixed or skipped and recorded in
    plan.warnings, or raised in strict mode. Warnings about an operation
    start with "op N (feature)", counting from 1.
    """
    plan = TTSPlan(text=base_text)

    clean, removed = sanitize_text(base_text)
    if removed:
        message = f"base_text: removed markup {', '.join(removed)}"
        if strict:
            raise ValueError(message)
        plan.text = clean
        _warn(plan, message)

    for number, op in enumerate(operations, start=1):
        feature = None
        try:
            feature = op.get("feature")
            kwargs = dict(op.get("kwargs") or {})

            # Lookup the function in the index
            func = PLAN_INDEX.get(feature) or FORMATTING_INDEX.get(feature)
            if func is None:
                raise ValueError("feature not found in Option Table")

            if "target" in kwargs:
                name = _target_param(func)
                if name is None or name in kwargs:
                    raise TypeError("'target' is not accepted here" if name is None
                                    else f"pass either 'target' or {name!r}, not both")
                kwargs[name] = kwargs.pop("target")
            for note in _clamp_kwargs(kwargs, strict):
                _warn(plan, f"op {number} ({feature}): {note}")

            if feature in PLAN_INDEX:
                previous = plan.instruct
                func(plan, **kwargs)
                if previous is not None and plan.instruct != previous:
                    _warn(plan, f"op {number} ({feature}): replaced earlier instruction {previous!r}")
            else:
                plan.text = func(plan.text, **kwargs)
        except Exception as e:
            if strict:
                raise
            _warn(plan, f"op {number} ({feature}): {e}")

    for note in _advisories(plan.text):
        _warn(plan, note)

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


def synthesize(cosyvoice, plan: TTSPlan, prompt_wav: str, prompt_text: str | None = None, **kwargs):
    """Synthesizes a plan with a loaded CosyVoice2 model, stitching audio and silence.

    Uses inference_instruct2 when the plan has an instruction, inference_zero_shot
    when prompt_text is given, and inference_cross_lingual otherwise. Extra
    kwargs are passed to the inference call. prompt_wav is a path to the
    reference audio file.

    text_frontend defaults to False: the frontend's sentence splitter collapses
    "..." into a single ".", which cancels the dramatic pause. With it off,
    numbers are not spelled out and long text is not split, so the plan gets
    an advisory for either. Pass text_frontend=True to keep the frontend.
    """
    if kwargs.pop("stream", False):
        if plan.speed != 1.0:
            raise ValueError(f"speed={plan.speed} requires stream=False; CosyVoice 2 cannot change speed while streaming")
        raise ValueError("synthesize returns one stitched tensor and cannot stream; call it with stream=False")

    # Imported here so the text logic stays importable without torch
    import torch

    instruct = plan.instruct
    if instruct:
        # The frontend does not add the separator itself
        if not instruct.endswith(END_OF_PROMPT):
            instruct += END_OF_PROMPT
        if prompt_text:
            _warn(plan, "synthesize: prompt_text ignored because the plan has an instruction")

    text_frontend = kwargs.setdefault("text_frontend", False)
    if text_frontend and SOFT_PAUSES["dramatic"] in plan.text:
        _warn(plan, "synthesize: text_frontend=True collapses '...' into '.', so the dramatic pause is lost; "
                    "leave text_frontend off, or use a breath or timed pause")

    kwargs.update(stream=False, speed=plan.speed)
    chunks = []
    for segment in split_on_pauses(plan.text):
        if isinstance(segment, TextSegment) and not text_frontend:
            plain = _strip_markup(segment.text)[0]
            if re.search(r'\d', plain):
                _warn(plan, f"synthesize: text segment {segment.text!r} contains digits, which are not "
                            "spelled out with text_frontend=False; write numbers as words")
            # CJK text has no spaces, so each character counts as a word
            words = len(_CJK_RE.sub(' ', plain).split()) + len(_CJK_RE.findall(plain))
            if words > FRONTEND_OFF_MAX_WORDS:
                _warn(plan, f"synthesize: a text segment has {words} words, which is not split into sentences "
                            f"with text_frontend=False; keep segments under about {FRONTEND_OFF_MAX_WORDS} words")
        if isinstance(segment, SilenceSegment):
            chunks.append(torch.zeros(1, int(cosyvoice.sample_rate * segment.ms / 1000)))
        elif instruct:
            outputs = cosyvoice.inference_instruct2(segment.text, instruct, prompt_wav, **kwargs)
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
