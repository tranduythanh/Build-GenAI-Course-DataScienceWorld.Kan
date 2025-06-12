import os
import tempfile
import unittest

from PIL import Image

# isort: off
from unit14.agents.tools import (
    analyze_picture,
    assess_description,
    generate_sample_description,
    provide_feedback,
    transcribe_audio,
)

# isort: on


class TestTools(unittest.TestCase):
    def setUp(self):
        fd, self.img_path = tempfile.mkstemp(suffix=".png")
        Image.new("RGB", (64, 64)).save(self.img_path)
        os.close(fd)

    def tearDown(self):
        os.remove(self.img_path)

    def test_transcribe_audio(self):
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        result = transcribe_audio.run(audio_path)
        self.assertIsInstance(result, str)
        os.remove(audio_path)

    def test_pipeline(self):
        info = analyze_picture.run(self.img_path)
        sample = generate_sample_description.run(info)
        score = assess_description.run(sample, "A simple test description")
        feedback = provide_feedback.run(score)
        self.assertIsInstance(feedback, str)


if __name__ == "__main__":
    unittest.main()
