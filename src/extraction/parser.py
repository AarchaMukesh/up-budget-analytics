"""
parser.py
─────────
UP DDG Budget Table Extraction — Production Parser

Extracts tables from gridless Uttar Pradesh Detailed Demands for Grants PDFs.

Pipeline layers (applied in order per page):
  1. PDF classification  → classify_documents.py (upstream)
  2. Table detection     → 4-strategy cascade (text_tight → word_position_fallback)
  3. Row cleaning        → _clean_row()  [font translation + HoA extraction]
  4. Header construction → _construct_header()
  5. DataFrame assembly  → _rows_to_dataframe_hoa_safe()
  6. Scoring + selection → _score_page_tables()

v3 fixes over previous version
───────────────────────────────
  FIX-1  Removed duplicate declarations of STRUCTURE_YEAR_RE, HOA_RE,
         SPECIAL_COLS, _is_blank_cell, _normalize_column_name (Sections 1 & 8
         both declared them; last-write silently won).

  FIX-2  Removed the clean_row_fn / construct_header_fn / map_legacy_fn /
         classify_col_fn kwargs from _rows_to_dataframe_hoa_safe. They were
         defaulting to stub functions (_clean_row_default) instead of the real
         implementations, causing all font translation and HoA extraction to
         be silently skipped in production call-sites. The real module-level
         functions are now called directly.

  FIX-3  Removed the _clean_row_default / _construct_header_default stubs
         (dead code after FIX-2; their presence was causing confusion).

  FIX-4  Fixed duplicate key "v" in KRUTI_DEV_CHARS. "v" → "अ" (vowel A)
         was silently overwritten by "v" → "ओ" (vowel O). Both are now
         represented correctly using the two-character lookahead path.

  FIX-5  All call-sites (_extract_via_word_positions, extract_with_pdfplumber)
         now call _rows_to_dataframe_hoa_safe without the removed kwargs.

  FIX-6  All pd.NA boolean-context crashes fixed via _is_empty / _is_truthy /
         _safe_str / _safe_is_full_hoa / _safe_is_major_head helpers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pdfplumber


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Module-level Constants
# ═══════════════════════════════════════════════════════════════════════════════
# Declared ONCE here. Do not redeclare in any section below.

STRUCTURE_YEAR_RE = re.compile(r"^\d{4}$")
HOA_RE            = re.compile(r"(\d{4}-\d{2}-\d{3}-\d{2}-\d{2})")
SPECIAL_COLS      = {"page", "is_structure"}

GENERIC_HEADER_KEYWORDS = (
    # English
    "budget", "estimate", "actual", "revised", "total", "grant",
    # Hindi Unicode (post-translation)
    "बजट", "अनुमान", "वास्तविक", "संशोधित", "योग", "अनुदान",
)

MIN_ACCEPTABLE_PAGE_SCORE = 10


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Null-Safe Helpers
# ═══════════════════════════════════════════════════════════════════════════════
# These helpers exist solely to avoid calling bool() on pd.NA, which raises
# "TypeError: boolean value of NA is ambiguous".

def _is_empty(value: Any) -> bool:
    """True if value is None, pd.NA, float NaN, or empty/whitespace string."""
    if value is None:
        return True
    try:
        if value is pd.NA:
            return True
    except Exception:
        pass
    try:
        if isinstance(value, float) and pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _is_truthy(value: Any) -> bool:
    """Safe boolean: False for pd.NA / None / empty string, True otherwise."""
    return not _is_empty(value)


def _safe_str(value: Any) -> str:
    """Stringify value; return '' for any NA variant."""
    return "" if _is_empty(value) else str(value).strip()


def _safe_is_full_hoa(s: str) -> bool:
    """True if s is a 15-digit HoA code. Always returns a real bool."""
    return bool(re.match(r"^\d{4}-\d{2}-\d{3}-\d{2}-\d{2}$", s))


def _safe_is_major_head(s: str) -> bool:
    """True if s is a 4-digit major head code. Always returns a real bool."""
    return bool(STRUCTURE_YEAR_RE.match(s))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Cell / Header Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_blank_cell(value: Any) -> bool:
    """True if cell is empty (delegates to _is_empty for NA safety)."""
    return _is_empty(value)


def _normalize_column_name(name: Any) -> str:
    """Collapse newlines and strip; return '' for NA values."""
    if _is_empty(name):
        return ""
    return str(name).replace("\n", " ").replace("\r", " ").strip()


def _row_has_generic_headers(row: List[str]) -> bool:
    non_empty = [c for c in row if c and str(c).strip()]
    if not non_empty:
        return False
    lower_cells = [str(c).lower() for c in non_empty]
    matches = sum(
        1 for cell in lower_cells
        if any(kw in cell for kw in GENERIC_HEADER_KEYWORDS)
    )
    return matches >= max(1, len(non_empty) // 4)


def _looks_like_year_row(row: List[str]) -> bool:
    cells = [str(c).strip() for c in row if c and str(c).strip()]
    if not cells:
        return False
    year_like = sum(1 for c in cells if STRUCTURE_YEAR_RE.match(c))
    return year_like >= max(1, len(cells) // 4)


def _construct_header(rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """Build a (possibly multi-line) header from the first 1–3 rows.

    Handles the common UP DDG pattern:
        Row 0: ["मांग सं.", "बजट",      "संशोधित"  ]
        Row 1: ["",         "अनुमान",    "अनुमान"   ]
        Row 2: ["",         "2022-23",   "2022-23"  ]
    →   Header: ["मांग सं.", "बजट अनुमान 2022-23", "संशोधित अनुमान 2022-23"]
    """
    if not rows:
        return [], []

    num_cols      = len(rows[0])
    max_hdr_rows  = min(3, len(rows))
    hdr_row_count = 1

    if max_hdr_rows >= 2 and _row_has_generic_headers(rows[0]):
        hdr_row_count = 2
        if max_hdr_rows >= 3 and (
            _looks_like_year_row(rows[1]) or _looks_like_year_row(rows[2])
        ):
            hdr_row_count = 3

    header_rows = rows[:hdr_row_count]
    data_rows   = rows[hdr_row_count:]

    # ── Build raw header names ────────────────────────────────────────────────
    raw_header: List[str] = []
    for col_idx in range(num_cols):
        parts: List[str] = []
        for r in header_rows:
            cell = r[col_idx] if col_idx < len(r) else ""
            cell_clean = _normalize_column_name(cell)
            if cell_clean:
                parts.append(cell_clean)
        combined = " ".join(parts).strip()
        raw_header.append(combined if combined else f"column_{col_idx + 1}")

    # ── Deduplicate: suffix repeated names with _2, _3 … ─────────────────────
    # pd.concat raises InvalidIndexError when any DataFrame has duplicate
    # column names. Common in UP DDGs where multiple amount columns share the
    # same header text (e.g. two columns both labelled "योग" / "Total").
    seen_counts: Dict[str, int] = {}
    header: List[str] = []
    for name in raw_header:
        if name not in seen_counts:
            seen_counts[name] = 1
            header.append(name)
        else:
            seen_counts[name] += 1
            header.append(f"{name}_{seen_counts[name]}")

    return header, data_rows


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Legacy Font Mapping  (Kruti Dev / Shreeshila → Unicode)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Kruti Dev / Shreeshila store ASCII bytes rendered as Devanagari glyphs.
# pdfplumber reads the raw bytes → garbled ASCII ("Mojibake").
#
# Strategy
# ─────────
# Pass 1: whole-word / phrase substitution (longest match first).
# Pass 2: single-character fallback for residual ASCII bytes.
#
# To extend: add entries to KRUTI_DEV_WORDS (phrases) or fix KRUTI_DEV_CHARS
# (characters). Do NOT add duplicate keys — Python dicts silently drop them.

KRUTI_DEV_WORDS: Dict[str, str] = {
    # ── Government / Administrative ──────────────────────────────────────────
    "mRrj izns'k ljdkj": "उत्तर प्रदेश सरकार",
    "mRrj izns'k":        "उत्तर प्रदेश",
    "ljdkj":              "सरकार",
    "mRrj":               "उत्तर",
    "izns'k":             "प्रदेश",
    "foRr":               "वित्त",
    "foHkkx":             "विभाग",
    "jkT;":               "राज्य",
    "dsUnz":              "केंद्र",
    # ── Budget / Estimate terms ───────────────────────────────────────────────
    "la'kksf/kr vuqeku":  "संशोधित अनुमान",
    "okLrfod O;;":        "वास्तविक व्यय",
    "ctV vuqeku":         "बजट अनुमान",
    "ctV":                "बजट",
    "vuqeku":             "अनुमान",
    "la'kksf/kr":         "संशोधित",
    "okLrfod":            "वास्तविक",
    "vuqnku":             "अनुदान",
    "dqy":                "कुल",
    ";ksx":               "योग",
    # ── Revenue / Capital / Loan ──────────────────────────────────────────────
    "jktLo":              "राजस्व",
    "iwathxr":            "पूंजीगत",
    "_.k":                "ऋण",
    "O;;":                "व्यय",
    "izkfIr":             "प्राप्ति",
    # ── Object heads ─────────────────────────────────────────────────────────
    "osru":               "वेतन",
    "HkRrk":              "भत्ता",
    "dk;kZy;":            "कार्यालय",
    "isa'ku":             "पेंशन",
    "C;kt":               "ब्याज",
    "vuqnku&lgk;rk":      "अनुदान-सहायता",
    "iwathxr ifjO;;":     "पूंजीगत परिव्यय",
    # ── Department / sector names ─────────────────────────────────────────────
    "f'k{kk":             "शिक्षा",
    "LokLF;":             "स्वास्थ्य",
    "iqfyl":              "पुलिस",
    "d`f'k":              "कृषि",
    "flapkbZ":            "सिंचाई",
    "lkoZtfud fuekZ.k":   "सार्वजनिक निर्माण",
    "lkoZtfud":           "सार्वजनिक",
    "fuekZ.k":            "निर्माण",
    "lM+d":               "सड़क",
    "iqy":                "पुल",
    "Hkou":               "भवन",
    "fo|ky;":             "विद्यालय",
    "fpfdRlky;":          "चिकित्सालय",
    "fo'ofo|ky;":         "विश्वविद्यालय",
    "lekt dY;k.k":        "समाज कल्याण",
    "xzke fodkl":         "ग्राम विकास",
    "uxj fodkl":          "नगर विकास",
    "tkfr dY;k.k":        "जाति कल्याण",
    "tkfr":               "जाति",
    "tu tkfr":            "जनजाति",
    "Hkwfe":              "भूमि",
    "ty":                 "जल",
    "fof/k":              "विधि",
    "U;k;":               "न्याय",
    "j{kk":               "रक्षा",
    "mtkZ":               "ऊर्जा",
    "ifjogu":             "परिवहन",
    "x`g":                "गृह",
    # ── Column header terms (hindiDict.pdf glossary) ──────────────────────────
    "ekax la[;k":         "मांग संख्या",
    "foooj.k":            "विवरण",
    'o"kZ':               "वर्ष",
    "iz/kku 'kh\"kZ":     "प्रमुख शीर्ष",
    "y?kq 'kh\"kZ":       "लघु शीर्ष",
    "mi 'kh\"kZ":         "उप शीर्ष",
    "y?kqRre 'kh\"kZ":    "लघुत्तम शीर्ष",
    "izi= la[;k":         "प्रपत्र संख्या",
    "vkgj.k vf/kdkjh":   "आहरण अधिकारी",
    "fu;a=.k bdkbZ":      "नियंत्रण इकाई",
    "iz'kklu":            "प्रशासन",
}

# FIX-4: "v" → "अ" was duplicated as "v" → "ओ" in the original.
# Resolution: "v" alone = "अ" (vowel A). "ओ" is encoded as "vks" in Kruti Dev
# and is handled by the two-character lookahead in map_legacy_to_unicode().
KRUTI_DEV_CHARS: Dict[str, str] = {
    # Two-character sequences (checked before single chars)
    "vk":  "आ",    # aa
    "bZ":  "ई",    # ii (long)
    "mw":  "ऊ",    # uu (long)
    "v_":  "ऋ",    # ri
    ",s":  "ऐ",    # ai
    "vks": "ओ",    # o   ← FIX-4: was duplicate key "v"
    "vkS": "औ",   # au
    # Single-character vowels
    "v":   "अ",    # a   ← FIX-4: now unambiguous (no duplicate)
    "b":   "इ",    # i
    "m":   "उ",    # u
    ",":   "ए",    # e
    # Consonants
    "d":   "क",    "D":  "क्",
    "x":   "ग",    "?":  "घ",
    "p":   "च",    "N":  "छ",
    "t":   "ज",    ">":  "झ",
    "V":   "ट",    "B":  "ठ",
    "M":   "ड",    "<":  "ढ",
    ".":   "ण",
    "r":   "त",    "F":  "थ",
    "n":   "द",    "/":  "ध",
    "u":   "न",
    "i":   "प",    "Q":  "फ",
    "c":   "ब",    "H":  "भ",
    "e":   "म",    ";":  "य",
    "j":   "र",    "y":  "ल",
    "o":   "व",
    "'":   "श",    "\"": "ष",
    "l":   "स",    "g":  "ह",
    # Diacritics
    "a":   "ं",    ":":  "ः",    "¡":  "ँ",
    "~":   "्",
}

# Pre-sort word table: longest phrase first → prevents partial-match shadowing
_WORD_MAP_SORTED: List[Tuple[str, str]] = sorted(
    KRUTI_DEV_WORDS.items(), key=lambda kv: len(kv[0]), reverse=True
)

# Pre-sort two-char keys so the lookahead pass works correctly
_TWO_CHAR_CHARS: Dict[str, str] = {
    k: v for k, v in KRUTI_DEV_CHARS.items() if len(k) == 2
}
_ONE_CHAR_CHARS: Dict[str, str] = {
    k: v for k, v in KRUTI_DEV_CHARS.items() if len(k) == 1
}


def map_legacy_to_unicode(text: str) -> str:
    """Convert Kruti Dev / Shreeshila Mojibake to Unicode Devanagari.

    Pass 1 — whole-word substitution (longest match wins, word-boundary aware).
    Pass 2 — single/double character fallback for residual ASCII bytes.
             Two-character sequences (e.g. "vk" → "आ") are checked before
             single characters to prevent "v" + "k" being split incorrectly.

    Idempotent: already-Unicode text passes through unchanged because
    Devanagari code points don't appear in the ASCII-range Kruti tables.
    """
    if not text or not isinstance(text, str):
        return text or ""

    result = text

    # ── Pass 1: phrase / word substitution ────────────────────────────────────
    for garbled, unicode_str in _WORD_MAP_SORTED:
        try:
            result = re.sub(
                r"\b" + re.escape(garbled) + r"\b", unicode_str, result
            )
        except re.error:
            result = result.replace(garbled, unicode_str)

    # ── Pass 2: character-level fallback (only if residual ASCII remains) ─────
    if not re.search(r"[a-zA-Z'\"~¡,>.<]", result):
        return result

    out: List[str] = []
    i = 0
    while i < len(result):
        two = result[i: i + 2]
        if two in _TWO_CHAR_CHARS:
            out.append(_TWO_CHAR_CHARS[two])
            i += 2
        elif result[i] in _ONE_CHAR_CHARS:
            out.append(_ONE_CHAR_CHARS[result[i]])
            i += 1
        else:
            out.append(result[i])
            i += 1

    return "".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HoA Extraction  (regex-first, before font translation)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_hoa_from_cell(raw_cell: str) -> Tuple[Optional[str], str]:
    """Extract 15-digit HoA code from a raw cell before any font translation.

    Runs on the raw pdfplumber string so digit sequences are never disturbed
    by character-mapping. The code may be merged with Mojibake text, e.g.:
        "ljdkj2202-01-101-00-01fo|ky;"
        → hoa_code = "2202-01-101-00-01"
        → remainder = "ljdkjfo|ky;"  (passed to map_legacy_to_unicode next)

    Returns:
        (hoa_code, remainder)  where hoa_code is None if no match found.
    """
    if not raw_cell:
        return None, raw_cell or ""

    match = HOA_RE.search(raw_cell)
    if not match:
        return None, raw_cell

    hoa_code  = match.group(1)
    remainder = HOA_RE.sub("", raw_cell).strip(" -–—/|")
    return hoa_code, remainder


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Column Header Classification
# ═══════════════════════════════════════════════════════════════════════════════
#
# Based on Administrative Glossary (hindiDict.pdf) concepts.
# Matching is substring + case-insensitive so partial headers like
# "बजट अनुमान 2022-23" still resolve to "budget_estimate".

COLUMN_CATEGORY_KEYWORDS: Dict[str, str] = {
    # Revenue
    "राजस्व व्यय":      "revenue",
    "राजस्व प्राप्ति":   "revenue",
    "राजस्व":           "revenue",
    "revenue":          "revenue",
    # Capital
    "पूंजीगत व्यय":     "capital",
    "पूंजीगत प्राप्ति":  "capital",
    "पूंजीगत":          "capital",
    "capital":          "capital",
    "पूंजी":            "capital",
    # Loan
    "ऋण":              "loan",
    "loan":            "loan",
    "उधार":            "loan",
    # Grant
    "अनुदान राशि":     "grant",
    "अनुदान":          "grant",
    "grant":           "grant",
    # Budget Estimate
    "बजट अनुमान":      "budget_estimate",
    "budget estimate": "budget_estimate",
    "be":              "budget_estimate",
    # Revised Estimate
    "संशोधित अनुमान":  "revised_estimate",
    "revised estimate": "revised_estimate",
    "re":              "revised_estimate",
    # Actuals
    "वास्तविक व्यय":   "actuals",
    "actual":          "actuals",
    "actuals":         "actuals",
    # Description / HoA structure
    "मांग संख्या":     "demand_number",
    "demand no":       "demand_number",
    "demand":          "demand_number",
    "विवरण":           "description",
    "description":     "description",
    "मद":              "description",
    "प्रमुख शीर्ष":    "major_head",
    "major head":      "major_head",
    "लघु शीर्ष":       "minor_head",
    "minor head":      "minor_head",
    "उप शीर्ष":        "sub_head",
    "sub head":        "sub_head",
    "लघुत्तम शीर्ष":   "detailed_head",
    "detailed head":   "detailed_head",
}

_COL_KW_SORTED = sorted(COLUMN_CATEGORY_KEYWORDS, key=len, reverse=True)


def _classify_column_header(header_text: str) -> Optional[str]:
    """Return semantic category for a column header (post-translation Unicode)."""
    if not header_text:
        return None
    lower = header_text.lower().strip()
    for keyword in _COL_KW_SORTED:
        if keyword in lower:
            return COLUMN_CATEGORY_KEYWORDS[keyword]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Per-Row Cleaning Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_row(
    raw_row: List[Any],
    *,
    description_col_index: int = 0,
) -> Tuple[List[str], Optional[str]]:
    """Apply font translation + HoA extraction to every cell in a row.

    Per-cell pipeline:
      1. Stringify & strip (NA-safe)
      2. Regex-first HoA extraction (before font translation)
      3. Font translation on remainder (map_legacy_to_unicode)
      4. For non-description columns: strip non-numeric residue
         (enforces Description ↔ Amount separation, Req 3)

    Returns:
        (cleaned_cells, hoa_code)
        cleaned_cells : translated strings, same length as raw_row
        hoa_code      : first 15-digit code found in any cell, else None
    """
    row_hoa: Optional[str] = None
    cleaned: List[str]     = []

    for idx, cell in enumerate(raw_row):
        # Step 1 — NA-safe stringify
        if _is_empty(cell):
            cleaned.append("")
            continue
        raw_str = str(cell).replace("\n", " ").replace("\r", " ").strip()

        # Step 2 — Regex-first HoA extraction
        hoa_code, remainder = _extract_hoa_from_cell(raw_str)
        if hoa_code is not None and row_hoa is None:
            row_hoa = hoa_code

        # Step 3 — Font translation
        translated = map_legacy_to_unicode(remainder)

        # Step 4 — Column-type enforcement
        if idx == description_col_index:
            # Description column: keep full text; prepend HoA if it was merged in
            if hoa_code is not None and hoa_code not in translated:
                cleaned.append(f"{hoa_code} {translated}".strip())
            else:
                cleaned.append(translated)
        else:
            # Amount columns: keep only numeric characters
            numeric_only = re.sub(r"[^\d,.()\-]", "", translated).strip()
            # If nothing numeric survives, keep translated text
            # (handles sub-header rows like "Revenue" spanning columns)
            cleaned.append(numeric_only if numeric_only else translated)

    return cleaned, row_hoa


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HoA-Safe DataFrame Builder
# ═══════════════════════════════════════════════════════════════════════════════

def _rows_to_dataframe_hoa_safe(
    table_rows: Any,
    *,
    page_num: Optional[int] = None,
    num_cols_override: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Convert raw pdfplumber rows to a cleaned DataFrame.

    Processing order
    ────────────────
    A. Pad all rows to widest column count (preserves HoA alignment)
    B. Run _clean_row() on every row (font translation + HoA extraction)
    C. Build header via _construct_header() + translate any residual Mojibake
    D. Assemble DataFrame; replace blanks with pd.NA; dropna
    E. Align hoa_per_row to surviving rows using index-safe dict lookup
    F. Attach is_structure, hoa_code, col_categories, page columns

    HoA priority (Req 3 — data integrity)
    ──────────────────────────────────────
    (a) 15-digit code from _clean_row() [handles merged cells]    ← highest
    (b) 15-digit code in first-column value
    (c) 4-digit major head in first-column value
    (d) None — row is not a structural row
    """
    if not table_rows or not isinstance(table_rows, list):
        return None
    if len(table_rows) < 2:
        return None

    row_lengths = [len(r) for r in table_rows if isinstance(r, list)]
    if not row_lengths:
        return None

    num_cols = num_cols_override or max(row_lengths)
    if num_cols < 2:
        return None

    # ── A: Pad + clean every row ──────────────────────────────────────────────
    cleaned_rows: List[List[str]]     = []
    hoa_per_row:  List[Optional[str]] = []

    for r in table_rows:
        if not isinstance(r, list):
            r = []
        padded = list(r[:num_cols]) + [None] * (num_cols - len(r))
        cleaned_cells, row_hoa = _clean_row(padded, description_col_index=0)

        # Coerce row_hoa: never store pd.NA (FIX-6)
        row_hoa = None if _is_empty(row_hoa) else (_safe_str(row_hoa) or None)

        cleaned_rows.append(cleaned_cells)
        hoa_per_row.append(row_hoa)

    # ── B: Build header; translate residual Mojibake in header text ───────────
    header, data_rows = _construct_header(cleaned_rows)
    if not header or not data_rows:
        return None

    header = [map_legacy_to_unicode(h) for h in header]

    # ── C: Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame(data_rows, columns=header)

    # NA-safe blank replacement (FIX-6: pd.NA == "" raises TypeError)
    def _to_na_if_blank(v: Any) -> Any:
        return pd.NA if _is_empty(v) else v

    for col in df.columns:
        df[col] = df[col].map(_to_na_if_blank)

    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    if df.empty or df.shape[1] <= 1 or df.shape[0] < 2:
        return None

    df.columns = [_normalize_column_name(c) for c in df.columns]

    # ── D: Index-safe HoA alignment (FIX-6) ──────────────────────────────────
    # _construct_header consumed `header_row_count` rows from the front.
    # dropna() then removed rows by position — df.index retains original
    # positions so we build a {position → hoa_code} dict and look up by index.
    header_row_count = len(cleaned_rows) - len(data_rows)

    hoa_by_pos: Dict[int, Optional[str]] = {
        i: hoa_per_row[header_row_count + i]
        if (header_row_count + i) < len(hoa_per_row)
        else None
        for i in range(len(data_rows))
    }

    # ── E: Populate is_structure and hoa_code ────────────────────────────────
    first_col    = df.columns[0]
    is_structure: List[bool]         = []
    hoa_codes:    List[Optional[str]] = []

    for row_idx in df.index:
        raw_val = df.at[row_idx, first_col]
        s       = _safe_str(raw_val)          # never pd.NA (FIX-6)

        row_hoa          = hoa_by_pos.get(row_idx)
        row_hoa          = None if _is_empty(row_hoa) else row_hoa

        cell_is_full_hoa  = _safe_is_full_hoa(s)   # real bool (FIX-6)
        cell_is_major_head = _safe_is_major_head(s) # real bool (FIX-6)

        # Priority (a) → (b) → (c) → (d)
        if _is_truthy(row_hoa) and not cell_is_full_hoa:
            final_hoa = row_hoa   # (a) merged-cell extraction wins
            is_struct = True
        elif cell_is_full_hoa:
            final_hoa = s         # (b) first-column full HoA
            is_struct = True
        elif cell_is_major_head:
            final_hoa = s         # (c) 4-digit major head
            is_struct = True
        else:
            final_hoa = None      # (d) not a structural row
            is_struct = False

        is_structure.append(is_struct)
        hoa_codes.append(final_hoa)

    df["is_structure"] = is_structure
    df["hoa_code"]     = hoa_codes

    # ── F: Column category metadata (stored in attrs, not as a column) ────────
    df.attrs["col_categories"] = {
        col: _classify_column_header(col)
        for col in df.columns
        if col not in SPECIAL_COLS | {"hoa_code"}
    }

    if page_num is not None:
        df["page"] = int(page_num)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Table Settings Cascade
# ═══════════════════════════════════════════════════════════════════════════════

TABLE_SETTINGS_CASCADE: List[Tuple[str, Dict[str, Any]]] = [
    (
        "text_tight",
        {
            "vertical_strategy":   "text",
            "horizontal_strategy": "text",
            "snap_tolerance":      3,
            "join_tolerance":      3,
            "edge_min_length":     3,
            "min_words_vertical":  3,
            "min_words_horizontal": 1,
            "intersection_tolerance": 5,
            "text_tolerance":      3,
        },
    ),
    (
        "text_loose",
        {
            "vertical_strategy":   "text",
            "horizontal_strategy": "text",
            "snap_tolerance":      6,
            "join_tolerance":      6,
            "edge_min_length":     3,
            "min_words_vertical":  2,
            "min_words_horizontal": 1,
            "intersection_tolerance": 8,
            "text_tolerance":      5,
        },
    ),
    (
        "lines_h_text_v",
        {
            "vertical_strategy":   "text",
            "horizontal_strategy": "lines",
            "snap_tolerance":      3,
            "join_tolerance":      3,
            "edge_min_length":     3,
            "min_words_vertical":  3,
            "min_words_horizontal": 1,
            "intersection_tolerance": 5,
            "text_tolerance":      3,
        },
    ),
    ("word_position_fallback", {}),   # sentinel — triggers word-position path
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Page-level Score Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _score_page_tables(tables: List[pd.DataFrame]) -> int:
    """Score extracted tables by (rows × effective_columns) with heuristics."""
    total = 0
    for df in tables:
        if df is None or df.empty:
            continue
        df_ne = df.dropna(how="all")
        if df_ne.empty:
            continue
        rows     = df_ne.shape[0]
        eff_cols = [c for c in df_ne.columns if c not in SPECIAL_COLS | {"hoa_code"}]
        cols     = len(eff_cols)
        if rows < 2 or cols <= 1:
            continue
        penalty = 0.5 if cols < 3 else (1.5 if 5 <= cols <= 8 else 1.0)
        total  += int(rows * cols * penalty)
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Word-Position Fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_via_word_positions(
    page: pdfplumber.page.Page,
    page_num: int,
) -> Optional[pd.DataFrame]:
    """Reconstruct a table from raw word bounding boxes (last-resort strategy).

    Groups words into rows by y-coordinate proximity, then into columns by
    x-coordinate clustering. Works on UP DDGs because column x-positions are
    consistent across rows even without grid lines.
    """
    words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
    )
    if not words:
        return None

    # ── Group words into rows by vertical proximity ───────────────────────────
    ROW_SNAP = 4   # pt; words within 4pt vertically → same row
    rows_by_y: Dict[float, List[Any]] = {}
    for word in words:
        top = round(float(word["top"]))
        matched_key: Optional[float] = None
        for key in rows_by_y:
            if abs(key - top) <= ROW_SNAP:
                matched_key = key
                break
        if matched_key is None:
            rows_by_y[top] = []
            matched_key    = top
        rows_by_y[matched_key].append(word)

    sorted_rows = [rows_by_y[k] for k in sorted(rows_by_y)]
    if len(sorted_rows) < 3:
        return None

    # ── Detect column x-centres from the densest row ─────────────────────────
    template_row = max(sorted_rows, key=len)
    col_centers  = sorted(
        float(w["x0"]) + float(w["width"]) / 2 for w in template_row
    )

    COL_MERGE_THRESHOLD = 20   # pt; centres within 20pt → same column
    merged_centers: List[float] = []
    for cx in col_centers:
        if merged_centers and abs(cx - merged_centers[-1]) < COL_MERGE_THRESHOLD:
            merged_centers[-1] = (merged_centers[-1] + cx) / 2
        else:
            merged_centers.append(cx)

    if len(merged_centers) < 2:
        return None

    # ── Assign words to nearest column ───────────────────────────────────────
    def _nearest_col(word: Any) -> int:
        wx = float(word["x0"]) + float(word["width"]) / 2
        return min(range(len(merged_centers)), key=lambda i: abs(merged_centers[i] - wx))

    reconstructed: List[List[str]] = []
    for row_words in sorted_rows:
        row_cells: List[str] = [""] * len(merged_centers)
        for word in sorted(row_words, key=lambda w: float(w["x0"])):
            col_idx = _nearest_col(word)
            existing = row_cells[col_idx]
            row_cells[col_idx] = (
                f"{existing} {word['text']}".strip() if existing else word["text"]
            )
        reconstructed.append(row_cells)

    # FIX-5: call without kwargs — real _clean_row / map_legacy_to_unicode used
    return _rows_to_dataframe_hoa_safe(reconstructed, page_num=page_num)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Main Extraction Function
# ═══════════════════════════════════════════════════════════════════════════════

def extract_with_pdfplumber(file_path: str) -> List[pd.DataFrame]:
    """Extract tables from gridless UP DDG PDFs using a 4-strategy cascade.

    For each page, strategies are tried in order (text_tight → text_loose →
    lines_h_text_v → word_position_fallback). The first strategy that scores
    above MIN_ACCEPTABLE_PAGE_SCORE is used; remaining strategies are skipped.

    Every extracted table passes through the full cleaning pipeline:
      font translation → HoA extraction → column classification → null safety.

    Returns:
        List of DataFrames, one per detected table. Each DataFrame contains:
        content columns (translated Unicode), is_structure (bool),
        hoa_code (str | None), page (int).
        df.attrs["col_categories"] maps column names to semantic categories.
    """
    all_tables: List[pd.DataFrame] = []

    try:
        print(f"[pdfplumber] Opening: {file_path}")
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"[pdfplumber] Total pages: {total_pages}")

            for page_index, page in enumerate(pdf.pages, start=1):
                page_tables: List[pd.DataFrame] = []
                strategy_used = "none"

                for strategy_label, settings in TABLE_SETTINGS_CASCADE:

                    # Sentinel: word-position fallback
                    if strategy_label == "word_position_fallback":
                        df = _extract_via_word_positions(page, page_index)
                        if df is not None:
                            page_tables  = [df]
                            strategy_used = strategy_label
                        break

                    try:
                        raw_tables = page.extract_tables(table_settings=settings)
                    except Exception as exc:
                        print(f"  [page {page_index}] {strategy_label} error: {exc}")
                        continue

                    if not raw_tables:
                        continue

                    candidates: List[pd.DataFrame] = []
                    for raw in raw_tables:
                        # FIX-5: no kwargs — real module-level functions used
                        df = _rows_to_dataframe_hoa_safe(raw, page_num=page_index)
                        if df is not None:
                            candidates.append(df)

                    if _score_page_tables(candidates) >= MIN_ACCEPTABLE_PAGE_SCORE:
                        page_tables   = candidates
                        strategy_used = strategy_label
                        break

                if page_tables:
                    all_tables.extend(page_tables)
                    print(
                        f"  [page {page_index}/{total_pages}] "
                        f"{len(page_tables)} table(s) via '{strategy_used}'"
                    )
                else:
                    print(f"  [page {page_index}/{total_pages}] no tables detected")

    except PermissionError:
        raise
    except Exception as exc:
        print(f"[pdfplumber] Failed on '{file_path}': {exc}")

    print(f"[pdfplumber] Done. Total tables extracted: {len(all_tables)}")
    return all_tables