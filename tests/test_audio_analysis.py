"""Unit tests for audio_analysis: RMS timelines and hype-keyword detection.

WAV fixtures are generated in-memory with the stdlib wave module.
"""

from __future__ import annotations

import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from audio_analysis import (
        _hype_hits_in_text,
        _mono16_wav_rms_timeline,
        _summarize_rms_spikes,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # numpy missing
    _IMPORT_ERROR = exc


def _write_wav(path: Path, samples: list[int], framerate: int = 16000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


@unittest.skipIf(_IMPORT_ERROR is not None, f"optional deps missing: {_IMPORT_ERROR}")
class RmsTimelineTests(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        centers, rms, sr = _mono16_wav_rms_timeline(Path("does/not/exist.wav"))
        self.assertEqual((centers, rms, sr), ([], [], 0.0))

    def test_silence_has_near_zero_rms(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "silence.wav"
            _write_wav(p, [0] * 16000)  # 1 s of silence
            centers, rms, sr = _mono16_wav_rms_timeline(p, window_sec=0.5)
        self.assertEqual(sr, 16000.0)
        self.assertTrue(centers)
        self.assertTrue(all(v == 0.0 for v in rms))

    def test_loud_tone_has_high_rms_and_ordered_centers(self):
        n = 16000
        tone = [int(20000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tone.wav"
            _write_wav(p, tone)
            centers, rms, _sr = _mono16_wav_rms_timeline(p, window_sec=0.25)
        self.assertEqual(centers, sorted(centers))
        self.assertTrue(all(v > 0.3 for v in rms))

    def test_stereo_is_downmixed(self):
        # L = tone, R = inverted tone → mono mean cancels to silence
        n = 8000
        frames: list[int] = []
        for i in range(n):
            v = int(15000 * math.sin(2 * math.pi * 220 * i / 16000))
            frames += [v, -v]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "stereo.wav"
            _write_wav(p, frames, channels=2)
            _centers, rms, _sr = _mono16_wav_rms_timeline(p, window_sec=0.25)
        self.assertTrue(rms)
        self.assertTrue(all(v < 0.01 for v in rms))


@unittest.skipIf(_IMPORT_ERROR is not None, f"optional deps missing: {_IMPORT_ERROR}")
class SummarizeRmsSpikesTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(_summarize_rms_spikes([], []), "rms_windows=unavailable")

    def test_mismatched_lengths(self):
        self.assertEqual(_summarize_rms_spikes([1.0], [0.1, 0.2]), "rms_windows=unavailable")

    def test_spike_is_reported(self):
        centers = [float(i) for i in range(20)]
        rms = [0.05] * 19 + [0.9]
        line = _summarize_rms_spikes(centers, rms)
        self.assertIn("loud_windows", line)
        self.assertIn("global_peak 19.00s", line)

    def test_flat_signal_reports_no_spikes(self):
        centers = [float(i) for i in range(10)]
        line = _summarize_rms_spikes(centers, [0.1] * 10)
        self.assertIn("no windows above adaptive threshold", line)


@unittest.skipIf(_IMPORT_ERROR is not None, f"optional deps missing: {_IMPORT_ERROR}")
class HypeHitsTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(_hype_hits_in_text(""), [])
        self.assertEqual(_hype_hits_in_text("   "), [])

    def test_matches_are_case_insensitive(self):
        hits = _hype_hits_in_text("WHAT A PLAY, that was an INSANE clutch!")
        self.assertEqual(len(hits), 3)

    def test_word_boundaries_respected(self):
        # 'acetone' must not match the \bace\b pattern
        self.assertEqual(_hype_hits_in_text("acetone paraclutches"), [])

    def test_one_v_x_patterns(self):
        self.assertTrue(_hype_hits_in_text("he's in a 1v4 right now"))

    def test_no_duplicate_patterns(self):
        hits = _hype_hits_in_text("clutch clutch clutch")
        self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
