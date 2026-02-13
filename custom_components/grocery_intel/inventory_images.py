"""Inventory images (fridge/pantry/cupboard) ingestion + vision evidence."""
from __future__ import annotations

import base64
import hashlib
import logging
import json
import io
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from .const import (
    CONF_LLM_PROVIDER,
    CONF_LLM_MODEL,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_EXTRA_INSTRUCTIONS,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_EXTRA_INSTRUCTIONS,
)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
HEIC_EXTENSIONS = {".heic", ".heif"}

_LOGGER = logging.getLogger(__name__)


def _fingerprint_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _read_file_base64_sync(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in HEIC_EXTENSIONS:
        converted = _convert_heic_to_jpeg_bytes_sync(path)
        if not converted:
            return ""
        return base64.b64encode(converted).decode("ascii")
    return base64.b64encode(_read_bytes(path)).decode("ascii")


def _parse_exif_datetime(value: str, offset: str | None) -> datetime | None:
    """Parse EXIF date strings like '2026:02:04 18:40:00' with optional offset '+01:00'."""
    if not value:
        return None
    try:
        dt = datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None

    tz = None
    if offset:
        m = re.match(r"^\s*([+-])(\d{2}):?(\d{2})\s*$", str(offset))
        if m:
            sign = 1 if m.group(1) == "+" else -1
            hours = int(m.group(2))
            minutes = int(m.group(3))
            tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
    if tz is None:
        tz = dt_util.DEFAULT_TIME_ZONE
    return dt.replace(tzinfo=tz)


def _extract_taken_at_iso_sync(path: str) -> str | None:
    """Best-effort extraction of capture time from image metadata (EXIF).

    Returns an ISO string with timezone when possible. If no metadata is
    available or parsing fails, returns None.
    """
    try:
        try:
            from pillow_heif import register_heif_opener  # type: ignore

            register_heif_opener()
        except Exception:
            pass
        from PIL import Image  # type: ignore

        img = Image.open(path)
        exif = getattr(img, "getexif", None)
        if not callable(exif):
            return None
        exif_data = img.getexif()  # type: ignore[no-untyped-call]
        if not exif_data:
            return None

        # Common EXIF tags:
        # 36867 DateTimeOriginal, 36868 DateTimeDigitized, 306 DateTime.
        dt_str = exif_data.get(36867) or exif_data.get(36868) or exif_data.get(306)
        if not isinstance(dt_str, str) or not dt_str.strip():
            return None

        # OffsetTimeOriginal (36880) / OffsetTimeDigitized (36881) / OffsetTime (36882).
        offset = exif_data.get(36880) or exif_data.get(36881) or exif_data.get(36882)
        if not isinstance(offset, str):
            offset = None

        dt = _parse_exif_datetime(dt_str, offset)
        return dt.isoformat() if dt else None
    except Exception:
        return None


def _convert_heic_to_jpeg_bytes_sync(path: str) -> bytes:
    """Best-effort HEIC/HEIF -> JPEG conversion without hard dependencies."""
    try:
        try:
            from pillow_heif import register_heif_opener  # type: ignore

            register_heif_opener()
        except Exception:
            pass

        from PIL import Image  # type: ignore

        img = Image.open(path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True)
        return buf.getvalue()
    except Exception:
        pass

    tool = shutil.which("magick") or shutil.which("convert") or shutil.which("heif-convert")
    if not tool:
        return b""

    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(
            prefix="grocery_intel_heic_", suffix=".jpg", delete=False
        )
        tmp_path = tmp.name
        tmp.close()

        base = os.path.basename(tool)
        if base == "heif-convert":
            cmd = [tool, path, tmp_path]
        elif base == "magick":
            cmd = [tool, path, "-auto-orient", "-strip", "-quality", "90", tmp_path]
        else:
            cmd = [tool, path, "-auto-orient", "-strip", "-quality", "90", tmp_path]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return b""
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception:
        return b""
    finally:
        if tmp is not None:
            try:
                os.remove(tmp.name)
            except OSError:
                pass


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _unique_archive_path(archive_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(archive_dir, filename)
    if not os.path.exists(candidate):
        return candidate
    suffix = uuid.uuid4().hex[:6]
    return os.path.join(archive_dir, f"{base}_{suffix}{ext}")


def _archive_duplicate_path(archive_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(archive_dir, f"{base}_duplicate{ext}")
    if not os.path.exists(candidate):
        return candidate
    suffix = uuid.uuid4().hex[:6]
    return os.path.join(archive_dir, f"{base}_duplicate_{suffix}{ext}")


def scan_inventory_images_inbox_sync(
    inbox_path: str,
    archive_path: str,
    processed_fingerprints: set[str],
    *,
    on_success: str = "archive",
) -> dict[str, Any]:
    """Scan inbox and move new inventory images to archive."""
    _ensure_dir(inbox_path)
    _ensure_dir(archive_path)

    imported: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen = set(processed_fingerprints)

    try:
        filenames = sorted(os.listdir(inbox_path))
    except FileNotFoundError:
        return {"imported": [], "duplicates": []}

    for filename in filenames:
        full = os.path.join(inbox_path, filename)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            continue

        taken_at = _extract_taken_at_iso_sync(full)
        try:
            content = _read_bytes(full)
        except Exception:
            continue
        fp = _fingerprint_bytes(content)

        if fp in seen:
            archived = _archive_duplicate_path(archive_path, filename)
            try:
                shutil.move(full, archived)
            except Exception:
                continue
            duplicates.append(
                {
                    "filename": filename,
                    "path": full,
                    "archived_path": archived,
                    "fingerprint": fp,
                    "taken_at": taken_at,
                }
            )
            continue
        seen.add(fp)

        if on_success == "delete":
            try:
                os.remove(full)
            except Exception:
                continue
            archived = None
        else:
            archived = _unique_archive_path(archive_path, filename)
            try:
                shutil.move(full, archived)
            except Exception:
                continue

        imported.append(
            {
                "filename": filename,
                "path": full,
                "archived_path": archived,
                "fingerprint": fp,
                "taken_at": taken_at,
            }
        )

    return {"imported": imported, "duplicates": duplicates}


def cleanup_inventory_images_archive_sync(archive_path: str, ttl_days: int) -> int:
    """Delete old inventory images from archive."""
    if ttl_days <= 0:
        return 0
    try:
        filenames = os.listdir(archive_path)
    except FileNotFoundError:
        return 0

    now = dt_util.now().timestamp()
    cutoff = now - (ttl_days * 86400)
    deleted = 0
    for name in filenames:
        full = os.path.join(archive_path, name)
        try:
            st = os.stat(full)
        except Exception:
            continue
        if st.st_mtime < cutoff:
            try:
                os.remove(full)
                deleted += 1
            except Exception:
                continue
    return deleted


def _join_url(base: str, path: str) -> str:
    return (base.rstrip("/") + "/" + path.lstrip("/")).rstrip("/")


def _extract_first_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start : end + 1].strip()
    try:
        parsed = json.loads(candidate)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_openai_output_text(payload: dict[str, Any]) -> str:
    out = payload.get("output_text")
    if isinstance(out, str) and out.strip():
        return out
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                chunks.append(txt)
    return "\n".join(chunks).strip()


def _normalize_detected_items(raw: Any) -> list[dict[str, Any]]:
    """Return list of detected items with label + confidence."""
    items: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for row in raw:
            label: str | None = None
            conf: float | None = None
            if isinstance(row, str):
                label = row.strip()
            elif isinstance(row, dict):
                label = str(row.get("label") or row.get("name") or "").strip()
                conf_raw = row.get("confidence")
                try:
                    conf = float(conf_raw) if conf_raw is not None else None
                except Exception:
                    conf = None
            if not label:
                continue
            if conf is None:
                conf = 0.7
            conf = max(0.0, min(1.0, conf))
            items.append({"label": label, "confidence": conf})
    elif isinstance(raw, dict):
        # Allow a dict mapping -> truthy values
        for k, v in raw.items():
            if not v:
                continue
            label = str(k).strip()
            if label:
                items.append({"label": label, "confidence": 0.7})
    return items[:100]


def _llm_inventory_system_prompt(extra: str = "") -> str:
    extra = (extra or "").strip()
    base = (
        "You are an assistant that analyzes a household inventory photo (fridge/pantry/cupboard). "
        "Return ONLY valid JSON (no markdown). "
        "Goal: list visible grocery items at a generic level.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "detected_items": [\n'
        '    {"label": "milk", "confidence": 0.0}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Use short generic labels (e.g., 'milk', 'eggs', 'banana', 'cheese').\n"
        "- Avoid brand names, sizes, and counts unless very obvious.\n"
        "- confidence is between 0 and 1.\n"
    )
    if extra:
        base += "\nExtra instructions:\n" + extra + "\n"
    return base


async def async_analyze_inventory_image(
    hass,
    *,
    entry,
    filename: str,
    image_b64: str,
) -> dict[str, Any]:
    llm_provider = (entry.options.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER) or "").strip()
    llm_model = (entry.options.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL) or "").strip()
    llm_api_key = (entry.options.get(CONF_LLM_API_KEY, DEFAULT_LLM_API_KEY) or "").strip()
    llm_base_url = (entry.options.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL) or "").strip()
    llm_extra = (entry.options.get(CONF_LLM_EXTRA_INSTRUCTIONS, DEFAULT_LLM_EXTRA_INSTRUCTIONS) or "").strip()

    provider = llm_provider.lower()
    if not llm_model:
        return {}

    session = async_get_clientsession(hass)
    system_prompt = _llm_inventory_system_prompt(llm_extra)

    if provider == "ollama":
        base = llm_base_url or "http://host.docker.internal:11434"
        url = _join_url(base, "/api/chat")
        payload = {
            "model": llm_model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Analyze inventory photo: {filename}",
                    "images": [image_b64],
                },
            ],
            "options": {"temperature": 0},
        }
        try:
            async with session.post(
                url,
                timeout=aiohttp.ClientTimeout(total=120),
                json=payload,
            ) as response:
                if response.status >= 400:
                    raise aiohttp.ClientError(f"HTTP {response.status}")
                data = await response.json()
        except Exception as err:
            _LOGGER.warning("LLM inventory analyzer (Ollama) failed for %s: %s", filename, err)
            return {}

        if not isinstance(data, dict):
            return {}
        msg = data.get("message", {})
        content = msg.get("content") if isinstance(msg, dict) else ""
        if not isinstance(content, str):
            return {}
        return _extract_first_json_object(content)

    if provider == "openai":
        if not llm_api_key:
            _LOGGER.warning("LLM inventory analyzer (OpenAI) missing API key; skipping")
            return {}
        ext = os.path.splitext(filename)[1].lower()
        if ext in {".png"}:
            mime = "image/png"
        elif ext in {".webp"}:
            mime = "image/webp"
        else:
            # HEIC/HEIF conversion produces JPEG bytes; JPEG is also a safe fallback.
            mime = "image/jpeg"
        base = llm_base_url or "https://api.openai.com"
        url = _join_url(base, "/v1/responses")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "detected_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["label", "confidence"],
                    },
                }
            },
            "required": ["detected_items"],
        }
        data_url = f"data:{mime};base64,{image_b64}"
        payload = {
            "model": llm_model,
            "input": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"filename: {filename}\nAnalyze inventory photo."},
                        {"type": "input_image", "image_url": data_url},
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "inventory_items",
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        try:
            async with session.post(
                url,
                timeout=aiohttp.ClientTimeout(total=120),
                headers={"Authorization": f"Bearer {llm_api_key}"},
                json=payload,
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(f"OpenAI HTTP {response.status}: {body[:500]}")
                data = await response.json()
        except Exception as err:
            _LOGGER.warning("LLM inventory analyzer (OpenAI) failed for %s: %s", filename, err)
            return {}
        if not isinstance(data, dict):
            return {}
        out_text = _extract_openai_output_text(data)
        return _extract_first_json_object(out_text)

    _LOGGER.warning("LLM inventory analyzer: unsupported provider=%s", llm_provider)
    return {}


def normalize_items_from_llm_result(result: dict[str, Any]) -> list[str]:
    detected = result.get("detected_items")
    return [str(r.get("label")) for r in _normalize_detected_items(detected) if r.get("label")]


def normalize_items_with_confidence_from_llm_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    detected = result.get("detected_items")
    return _normalize_detected_items(detected)
