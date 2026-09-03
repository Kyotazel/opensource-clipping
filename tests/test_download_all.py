"""Tests for zip-all clip download helper."""
import os
import tempfile
import unittest
from types import SimpleNamespace

from web.api.clip_zip import clip_files_for_zip


class ClipFilesForZipTest(unittest.TestCase):
    def test_skips_missing_and_unsafe_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "abc123"
            job_dir = os.path.join(tmp, job_id)
            os.makedirs(job_dir)
            with open(os.path.join(job_dir, "highlight_rank_1.mp4"), "wb") as f:
                f.write(b"x")
            clips = [
                {"filename": "highlight_rank_1.mp4"},
                {"filename": "missing.mp4"},
                {"filename": "../etc/passwd"},
            ]
            files = clip_files_for_zip(tmp, job_id, clips)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0][1], "highlight_rank_1.mp4")

    def test_accepts_objects_with_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "job9"
            job_dir = os.path.join(tmp, job_id)
            os.makedirs(job_dir)
            with open(os.path.join(job_dir, "clip.mp4"), "wb") as f:
                f.write(b"x")
            files = clip_files_for_zip(tmp, job_id, [SimpleNamespace(filename="clip.mp4")])
            self.assertEqual(files[0][1], "clip.mp4")

    def test_rejects_dotted_job_id(self):
        self.assertEqual(clip_files_for_zip("/tmp", "../x", [{"filename": "a.mp4"}]), [])


if __name__ == "__main__":
    unittest.main()
