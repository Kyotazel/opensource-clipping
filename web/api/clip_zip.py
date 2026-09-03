"""Collect rendered clip files for a download-all zip."""

from __future__ import annotations

import os


def clip_files_for_zip(outputs_dir: str, job_id: str, clips) -> list[tuple[str, str]]:
    """Return (absolute_path, zip_arcname) for each existing clip MP4."""
    if ".." in (job_id or ""):
        return []
    job_dir = os.path.join(outputs_dir, job_id)
    files: list[tuple[str, str]] = []
    used_names: set[str] = set()
    for clip in clips or []:
        filename = getattr(clip, "filename", None)
        if filename is None and isinstance(clip, dict):
            filename = clip.get("filename") or ""
        filename = os.path.basename(str(filename or ""))
        if not filename or filename in {".", ".."}:
            continue
        path = os.path.join(job_dir, filename)
        if not os.path.isfile(path):
            continue
        arcname = filename
        if arcname in used_names:
            stem, ext = os.path.splitext(filename)
            arcname = f"{stem}_{len(used_names)}{ext}"
        used_names.add(arcname)
        files.append((path, arcname))
    return files
