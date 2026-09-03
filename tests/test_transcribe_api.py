"""Tests for OpenAI-compatible STT response parsing."""
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from clipping.stt import (
    STT_FORMAT_ATTEMPTS,
    format_transkrip,
    segments_from_stt_response,
    transcribe_chunk_via_api,
    transkrip_has_text,
)


class SegmentsFromSttResponseTest(unittest.TestCase):
    def test_json_text_only_no_longer_yields_zero_segments(self):
        # This is the production bug: response_format=json returns {text} only.
        resp = {"text": "Halo dunia dari OpenRouter"}
        segs = segments_from_stt_response(resp, chunk_start=0.0, chunk_duration=12.5)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["words"][0]["word"], "Halo dunia dari OpenRouter")
        self.assertEqual(segs[0]["end"], 12.5)
        text = format_transkrip(segs)
        self.assertIn("Halo dunia dari OpenRouter", text)
        self.assertTrue(transkrip_has_text(segs))

    def test_empty_json_stays_empty(self):
        segs = segments_from_stt_response({"text": ""}, 0.0, 10.0)
        self.assertEqual(segs, [])
        self.assertFalse(transkrip_has_text(segs))
        self.assertEqual(format_transkrip(segs), "")

    def test_word_timestamps_group_by_max_words(self):
        resp = SimpleNamespace(
            text="satu dua tiga empat lima enam",
            words=[
                SimpleNamespace(word="satu", start=0.0, end=0.3),
                SimpleNamespace(word="dua", start=0.3, end=0.5),
                SimpleNamespace(word="tiga", start=0.5, end=0.8),
                SimpleNamespace(word="empat", start=0.8, end=1.1),
                SimpleNamespace(word="lima", start=1.1, end=1.4),
                SimpleNamespace(word="enam", start=1.4, end=1.8),
            ],
            segments=[],
        )
        segs = segments_from_stt_response(
            resp, chunk_start=600.0, chunk_duration=10.0, max_words_per_subtitle=5
        )
        self.assertEqual(len(segs), 2)
        self.assertEqual([w["word"] for w in segs[0]["words"]],
                         ["satu", "dua", "tiga", "empat", "lima"])
        self.assertEqual([w["word"] for w in segs[1]["words"]], ["enam"])
        self.assertAlmostEqual(segs[0]["start"], 600.0)
        self.assertAlmostEqual(segs[1]["start"], 601.4)

    def test_words_nested_under_segments(self):
        resp = {
            "text": "hello world",
            "words": None,
            "segments": [{
                "start": 1.0,
                "end": 2.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 1.0, "end": 1.4},
                    {"word": "world", "start": 1.4, "end": 2.0},
                ],
            }],
        }
        segs = segments_from_stt_response(resp, 0.0, 5.0, max_words_per_subtitle=5)
        self.assertEqual(len(segs), 1)
        self.assertEqual([w["word"] for w in segs[0]["words"]], ["hello", "world"])

    def test_segment_timestamps_without_words(self):
        resp = {
            "text": "kalimat satu. kalimat dua.",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "kalimat satu."},
                {"start": 1.5, "end": 3.0, "text": "kalimat dua."},
            ],
        }
        segs = segments_from_stt_response(resp, 10.0, 5.0)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["words"][0]["word"], "kalimat satu.")
        self.assertAlmostEqual(segs[0]["start"], 10.0)
        self.assertAlmostEqual(segs[1]["end"], 13.0)

    def test_pydantic_model_dump_fallback(self):
        class FakeModel:
            text = None
            words = None
            segments = None

            def model_dump(self):
                return {
                    "text": "dari dump",
                    "words": [{"word": "dari", "start": 0.0, "end": 0.2},
                              {"word": "dump", "start": 0.2, "end": 0.5}],
                }

        segs = segments_from_stt_response(FakeModel(), 0.0, 1.0)
        self.assertEqual([w["word"] for w in segs[0]["words"]], ["dari", "dump"])


class TranscribeChunkViaApiTest(unittest.TestCase):
    def test_prefers_verbose_json_with_word_timestamps(self):
        client = Mock()
        client.audio.transcriptions.create.return_value = {"text": "ok"}
        file_obj = io.BytesIO(b"fake-mp3")
        transcribe_chunk_via_api(client, file_obj, "openai/whisper-large-v3-turbo")
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"], "verbose_json")
        self.assertEqual(kwargs["timestamp_granularities"], ["word"])

    def test_falls_back_when_verbose_json_rejected(self):
        client = Mock()
        client.audio.transcriptions.create.side_effect = [
            Exception("400 verbose_json not supported"),
            Exception("400 still no"),
            {"text": "plain"},
        ]
        file_obj = io.BytesIO(b"fake-mp3")
        resp = transcribe_chunk_via_api(client, file_obj, "model")
        self.assertEqual(resp, {"text": "plain"})
        self.assertEqual(client.audio.transcriptions.create.call_count, 3)
        formats = [
            c.kwargs["response_format"]
            for c in client.audio.transcriptions.create.call_args_list
        ]
        self.assertEqual(formats, ["verbose_json", "verbose_json", "json"])
        self.assertEqual(
            [a["response_format"] for a in STT_FORMAT_ATTEMPTS],
            formats,
        )

    def test_raises_when_every_format_fails(self):
        client = Mock()
        client.audio.transcriptions.create.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError) as ctx:
            transcribe_chunk_via_api(client, io.BytesIO(b"x"), "model")
        self.assertIn("Semua format STT gagal", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
