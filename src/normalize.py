import re
import unicodedata


_TASHKEEL_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

_TATWEEL = "\u0640"

_ALEF_MAP = str.maketrans({
    "\u0622": "\u0627",  # آ
    "\u0623": "\u0627",  # أ
    "\u0625": "\u0627",  # إ
    "\u0671": "\u0627",  # ٱ
})

_YAA_MAP = str.maketrans({
    "\u0649": "\u064A",  # ى -> ي
    "\u06CC": "\u064A",  # ی -> ي
})

_DIGIT_MAP = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    "\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9",
    "01234567890123456789",
)

_WS_RE = re.compile(r"\s+")



def _unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _remove_invisible(text: str) -> str:
    """Drop null chars and invisible/control characters"""
    out = []
    for ch in text:
        if ch in "\n\t ":
            out.append(ch)
        elif unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cs", "Cn"):
            continue
        else:
            out.append(ch)
    return "".join(out)


def _remove_tatweel(text: str) -> str:
    return text.replace(_TATWEEL, "")


def _remove_tashkeel(text: str) -> str:
    return _TASHKEEL_RE.sub("", text)


def _normalize_alef(text: str) -> str:
    return text.translate(_ALEF_MAP)


def _normalize_yaa(text: str) -> str:
    return text.translate(_YAA_MAP)


def _normalize_digits(text: str) -> str:
    return text.translate(_DIGIT_MAP)


def _collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()



def norm(text: str) -> str:
    """Return the normalized form of `text`."""
    if not text:
        return ""
    text = _unicode_normalize(text)
    text = _remove_invisible(text)
    text = _remove_tatweel(text)
    text = _remove_tashkeel(text)
    text = _normalize_alef(text)
    text = _normalize_yaa(text)
    text = _normalize_digits(text)
    text = _collapse_whitespace(text)
    return text

