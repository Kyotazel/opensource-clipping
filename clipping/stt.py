"""Parse OpenAI-compatible speech-to-text responses into pipeline segments.

OpenRouter / OpenAI only attach word and segment timestamps when
``response_format="verbose_json"``. The default ``json`` body is
``{text, usage}`` — ``timestamp_granularities`` is ignored there, which is
how jobs used to finish with 0 segments and an empty AI prompt.
"""

# Prefer word timestamps, then segment timestamps, then plain json text.
STT_FORMAT_ATTEMPTS = (
    {"response_format": "verbose_json", "timestamp_granularities": ["word"]},
    {"response_format": "verbose_json"},
    {"response_format": "json"},
)


def stt_field(obj, name, default=None):
    """Read *name* from an SDK object, dict, or pydantic model."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    val = getattr(obj, name, default)
    if val is not default and val is not None:
        return val
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            data = dump()
        except Exception:
            data = None
        if isinstance(data, dict) and name in data and data[name] is not None:
            return data[name]
    return default


def stt_text(resp) -> str:
    return str(stt_field(resp, "text") or "").strip()


def stt_words(resp) -> list:
    words = list(stt_field(resp, "words") or [])
    if words:
        return words
    nested = []
    for seg in (stt_field(resp, "segments") or []):
        nested.extend(stt_field(seg, "words") or [])
    return nested


def format_transkrip(data_segmen: list[dict]) -> str:
    lines = []
    for seg in data_segmen:
        text = " ".join(w["word"] for w in seg.get("words", []) if w.get("word"))
        lines.append(f"[{seg['start']:.1f} - {seg['end']:.1f}] {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def transkrip_has_text(data_segmen: list[dict]) -> bool:
    return any(
        str(w.get("word") or "").strip()
        for seg in data_segmen
        for w in (seg.get("words") or [])
    )


def segments_from_stt_response(
    resp,
    chunk_start: float,
    chunk_duration: float,
    max_words_per_subtitle: int = 5,
) -> list[dict]:
    """Turn an STT API response into the pipeline's data_segmen shape.

    Preference: word timestamps → segment timestamps → full-text fallback.
    An empty list means the provider returned nothing usable.
    """
    usable_words = []
    for w in stt_words(resp):
        token = str(stt_field(w, "word") or "").strip()
        if not token:
            continue
        ws = chunk_start + float(stt_field(w, "start", 0.0) or 0.0)
        we = chunk_start + float(stt_field(w, "end", 0.0) or 0.0)
        if we < ws:
            we = ws
        usable_words.append({"word": token, "start": ws, "end": we})

    grouped: list[dict] = []
    chunk_words: list[dict] = []
    first_start = chunk_start
    for i, w in enumerate(usable_words):
        if not chunk_words:
            first_start = w["start"]
        chunk_words.append(w)
        if len(chunk_words) >= max_words_per_subtitle or i == len(usable_words) - 1:
            grouped.append({
                "start": first_start,
                "end": chunk_words[-1]["end"],
                "words": chunk_words,
            })
            chunk_words = []
    if grouped:
        return grouped

    for s in (stt_field(resp, "segments") or []):
        text = str(stt_field(s, "text") or "").strip()
        if not text:
            continue
        ss = chunk_start + float(stt_field(s, "start", 0.0) or 0.0)
        se = chunk_start + float(stt_field(s, "end", 0.0) or 0.0)
        if se < ss:
            se = ss
        grouped.append({
            "start": ss,
            "end": se,
            "words": [{"word": text, "start": ss, "end": se}],
        })
    if grouped:
        return grouped

    text = stt_text(resp)
    if not text:
        return []
    duration = float(chunk_duration or 0.0)
    end = chunk_start + (duration if duration > 0 else 0.0)
    print(
        "      ⚠️ API tidak mengembalikan timestamps — "
        "pakai teks penuh sebagai 1 segmen.",
        flush=True,
    )
    return [{
        "start": chunk_start,
        "end": end,
        "words": [{"word": text, "start": chunk_start, "end": end}],
    }]


def transcribe_chunk_via_api(client, file_obj, model: str):
    """Request verbose_json+words, then degrade until the provider accepts it."""
    last_err = None
    for kwargs in STT_FORMAT_ATTEMPTS:
        try:
            file_obj.seek(0)
            return client.audio.transcriptions.create(
                model=model, file=file_obj, **kwargs
            )
        except Exception as e:
            last_err = e
            fmt = kwargs.get("response_format")
            gran = kwargs.get("timestamp_granularities")
            extra = f" + {','.join(gran)}" if gran else ""
            print(f"      ⚠️ STT {fmt}{extra} gagal ({e})", flush=True)
    raise RuntimeError(f"Semua format STT gagal: {last_err}")
