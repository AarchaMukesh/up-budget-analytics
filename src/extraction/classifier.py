"""PDF document classification utilities.

This module classifies PDF files as:
- "digital": all pages contain meaningful text
- "scanned": no pages contain meaningful text
- "mixed": some pages contain meaningful text
- "error": file could not be read
"""

from __future__ import annotations

import re
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import pdfplumber

CID_PATTERN = re.compile(r"(cid:\d+)")
CID_RATIO_THRESHOLD = 0.3
IMAGE_DOMINANCE_THRESHOLD = 0.7
DEFAULT_TEXT_THRESHOLD = 50
POOR_OCR_WHITESPACE_RATIO_THRESHOLD = 0.45
SAMPLE_PAGE_INDICES = (0, 1, 10, 50, 100)


def _crop_page_content_region(page: pdfplumber.page.Page) -> pdfplumber.page.Page:
    """Crop out top/bottom page margins to reduce header/footer noise."""
    top_margin = page.height * 0.1
    bottom_margin = page.height * 0.1
    top = top_margin
    bottom = page.height - bottom_margin

    # Guard against very small pages where the crop region could invert.
    if bottom <= top:
        return page

    return page.crop((0, top, page.width, bottom))


def _is_image_dominant(page: pdfplumber.page.Page) -> bool:
    """Return whether a single image dominates the page area.

    Uses max single-image coverage ratio to avoid overcounting layered images.
    """
    page_area = float(page.width) * float(page.height)
    if page_area <= 0:
        return False

    max_image_area = 0.0
    for img in page.images:
        width = float(img.get("width", 0) or 0)
        height = float(img.get("height", 0) or 0)
        max_image_area = max(max_image_area, width * height)

    if max_image_area <= 0:
        return False

    return (max_image_area / page_area) > IMAGE_DOMINANCE_THRESHOLD


def _compute_cid_ratio(text: str | None) -> float:
    """Compute CID ratio = CID matches / max(len(text), 1)."""
    if text is None:
        return 0.0
    cid_matches = len(CID_PATTERN.findall(text))
    return cid_matches / max(len(text), 1)


def _compute_whitespace_ratio(text: str | None) -> float:
    """Compute whitespace ratio = number_of_spaces / max(len(text), 1)."""
    if text is None:
        return 0.0
    space_count = text.count(" ")
    return space_count / max(len(text), 1)


def _classify_page(page: pdfplumber.page.Page, text_threshold: int) -> str:
    """Classify a single page as ``digital`` or ``scanned``.

    Order of checks:
    1) Header/footer cropping
    2) Image dominance (overrides text logic)
    3) Text extraction + CID filtering + threshold check
    4) Poor OCR heuristic (high whitespace ratio)
    """
    cropped_page = _crop_page_content_region(page)

    if _is_image_dominant(cropped_page):
        return "scanned"

    extracted_text = cropped_page.extract_text()
    if extracted_text is None:
        return "scanned"

    stripped_text = extracted_text.strip()
    if len(stripped_text) <= text_threshold:
        return "scanned"

    cid_ratio = _compute_cid_ratio(extracted_text)
    if cid_ratio > CID_RATIO_THRESHOLD:
        return "scanned"

    whitespace_ratio = _compute_whitespace_ratio(extracted_text)
    if whitespace_ratio > POOR_OCR_WHITESPACE_RATIO_THRESHOLD:
        return "scanned"

    return "digital"


def _classify_page_with_metrics(
    page: pdfplumber.page.Page, text_threshold: int
) -> Tuple[str, float, float, bool]:
    """Classify a single page and return per-page metrics.

    Returns:
        (classification, cid_ratio, whitespace_ratio, image_dominant)
    """
    cropped_page = _crop_page_content_region(page)
    image_dominant = _is_image_dominant(cropped_page)
    extracted_text = cropped_page.extract_text()

    cid_ratio = _compute_cid_ratio(extracted_text)
    whitespace_ratio = _compute_whitespace_ratio(extracted_text)

    if image_dominant:
        return "scanned", cid_ratio, whitespace_ratio, True

    if extracted_text is None:
        return "scanned", cid_ratio, whitespace_ratio, False

    stripped_text = extracted_text.strip()
    if len(stripped_text) <= text_threshold:
        return "scanned", cid_ratio, whitespace_ratio, False

    if cid_ratio > CID_RATIO_THRESHOLD:
        return "scanned", cid_ratio, whitespace_ratio, False

    if whitespace_ratio > POOR_OCR_WHITESPACE_RATIO_THRESHOLD:
        return "scanned", cid_ratio, whitespace_ratio, False

    return "digital", cid_ratio, whitespace_ratio, False


def _classify_document_pages(pdf: pdfplumber.pdf.PDF, text_threshold: int) -> str:
    """Classify a PDF by combining sampled-page and full-scan strategies."""
    total_pages = len(pdf.pages)
    if total_pages == 0:
        return "scanned"

    sample_indices: List[int] = []
    for idx in SAMPLE_PAGE_INDICES:
        if 0 <= idx < total_pages and idx not in sample_indices:
            sample_indices.append(idx)

    sampled_page_classes = [_classify_page(pdf.pages[idx], text_threshold) for idx in sample_indices]
    if sampled_page_classes and len(set(sampled_page_classes)) == 1:
        return sampled_page_classes[0]

    digital_pages = 0
    scanned_pages = 0
    for page in pdf.pages:
        page_class = _classify_page(page, text_threshold)
        if page_class == "digital":
            digital_pages += 1
        else:
            scanned_pages += 1

    if digital_pages == total_pages:
        return "digital"
    if scanned_pages == total_pages:
        return "scanned"
    return "mixed"


def _page_has_meaningful_text(page: pdfplumber.page.Page, text_threshold: int) -> bool:
    """Return whether a page is classified as containing meaningful text.

    Meaningful text is defined as:
    - extracted text is not ``None`` after header/footer cropping
    - length of stripped text is greater than ``text_threshold``
    - CID-encoded noise ratio is not above threshold
    - page is not image-dominant
    """
    return _classify_page(page, text_threshold) == "digital"


def _classify_document_full_with_metrics(
    pdf: pdfplumber.pdf.PDF, text_threshold: int
) -> Tuple[str, int, int, float, float, int]:
    """Classify a full document and compute aggregate metrics."""
    total_pages = len(pdf.pages)
    if total_pages == 0:
        return "scanned", 0, 0, 0.0, 0.0, 0

    digital_pages = 0
    scanned_pages = 0
    cid_ratio_total = 0.0
    whitespace_ratio_total = 0.0
    image_dominant_pages = 0

    for page in pdf.pages:
        page_class, cid_ratio, whitespace_ratio, image_dominant = _classify_page_with_metrics(
            page, text_threshold
        )
        if page_class == "digital":
            digital_pages += 1
        else:
            scanned_pages += 1

        cid_ratio_total += cid_ratio
        whitespace_ratio_total += whitespace_ratio
        if image_dominant:
            image_dominant_pages += 1

    classification = (
        "digital"
        if digital_pages == total_pages
        else "scanned" if scanned_pages == total_pages else "mixed"
    )

    avg_cid_ratio = cid_ratio_total / total_pages
    avg_whitespace_ratio = whitespace_ratio_total / total_pages

    return (
        classification,
        digital_pages,
        scanned_pages,
        avg_cid_ratio,
        avg_whitespace_ratio,
        image_dominant_pages,
    )


def classify_pdf(
    file_path: str, text_threshold: int = 50, return_metadata: bool = False
) -> str | Dict[str, Any]:
    """Classify a PDF as ``digital``, ``scanned``, ``mixed``, or ``error``.

    Args:
        file_path: Path to the PDF file.
        text_threshold: Minimum stripped text length for a page to be treated
            as containing meaningful text.
        return_metadata: If True, return a structured metadata dictionary instead
            of only the classification string.

    Returns:
        If ``return_metadata`` is False:
        - ``"digital"`` if all pages contain meaningful text
        - ``"scanned"`` if no pages contain meaningful text
        - ``"mixed"`` if some pages contain meaningful text and others do not
        - ``"error"`` if the file cannot be read

        If ``return_metadata`` is True:
        - a dictionary containing classification, per-document page stats, and
          confidence metrics.
    """
    effective_threshold = max(text_threshold, 0)

    try:
        with pdfplumber.open(file_path) as pdf:
            filename = os.path.basename(file_path)

            if return_metadata:
                (
                    classification,
                    digital_pages,
                    scanned_pages,
                    avg_cid_ratio,
                    avg_whitespace_ratio,
                    image_dominant_pages,
                ) = _classify_document_full_with_metrics(pdf, effective_threshold)

                total_pages = len(pdf.pages)
                return {
                    "filename": filename,
                    "classification": classification,
                    "total_pages": total_pages,
                    "page_stats": {
                        "digital_pages": digital_pages,
                        "scanned_pages": scanned_pages,
                    },
                    "confidence_metrics": {
                        "avg_cid_ratio": avg_cid_ratio,
                        "avg_whitespace_ratio": avg_whitespace_ratio,
                        "image_dominant_pages": image_dominant_pages,
                    },
                }

            return _classify_document_pages(pdf, effective_threshold)
    except Exception as exc:
        print(f"[classify_pdf] Failed to read '{file_path}': {exc}")
        if not return_metadata:
            return "error"

        filename = os.path.basename(file_path)
        return {
            "filename": filename,
            "classification": "error",
            "total_pages": 0,
            "page_stats": {"digital_pages": 0, "scanned_pages": 0},
            "confidence_metrics": {
                "avg_cid_ratio": 0.0,
                "avg_whitespace_ratio": 0.0,
                "image_dominant_pages": 0,
            },
        }


def classify_pdf_debug(file_path: str) -> List[Tuple[int, bool]]:
    """Return page-level text presence information for debugging.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of ``(page_number, has_text)`` tuples using 1-based page numbers.
        ``has_text`` is computed with the default threshold logic used by
        :func:`classify_pdf` (threshold of 50).

        If the file cannot be read, an empty list is returned and a debug
        message is printed.
    """
    default_threshold = DEFAULT_TEXT_THRESHOLD

    try:
        with pdfplumber.open(file_path) as pdf:
            return [
                (page_index, _page_has_meaningful_text(page, default_threshold))
                for page_index, page in enumerate(pdf.pages, start=1)
            ]
    except Exception as exc:
        print(f"[classify_pdf_debug] Failed to read '{file_path}': {exc}")
        return []


def classify_batch(file_paths: List[str], max_workers: int = 4) -> Dict[str, str]:
    """Classify multiple PDF files concurrently.

    Args:
        file_paths: Paths to PDF files.
        max_workers: Maximum worker processes used for classification.

    Returns:
        A mapping of file path to document classification result.
    """
    if not file_paths:
        return {}

    worker_count = max(1, max_workers)
    results: Dict[str, str] = {}

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_path = {executor.submit(classify_pdf, path): path for path in file_paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[path] = future.result()
            except Exception as exc:
                print(f"[classify_batch] Failed to classify '{path}': {exc}")
                results[path] = "error"

    return results