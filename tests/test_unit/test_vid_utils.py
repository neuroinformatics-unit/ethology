import os
import unittest
from unittest.mock import MagicMock, patch

from ethology.video.utilities import (
    combine_images_to_video,
    compress_video,
    crop_video_spatial,
    crop_video_time,
    extract_frames,
    get_video_specs,
    print_video_specs,
)


class TestVideoUtils(unittest.TestCase):
    @patch("ethology.video.utilities.ffmpeg.input")
    def test_compress_video(self, mock_input):
        # Setup chain: input -> output -> run.
        mock_run = MagicMock()
        chain = MagicMock()
        chain.run.return_value = None
        mock_input.return_value.output.return_value = chain

        compress_video("input.mp4", "output.mp4", crf=25, preset="fast")
        mock_input.return_value.output.assert_called_with(
            "output.mp4", vcodec="libx264", crf=25, preset="fast"
        )
        chain.run.assert_called_once_with(quiet=True, overwrite_output=True)

    @patch("ethology.video.utilities.ffmpeg.probe")
    def test_get_video_specs(self, mock_probe):
        # Dummy probe data
        dummy_probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "r_frame_rate": "30/1",
                    "codec_name": "h264",
                    "width": 640,
                    "height": 480,
                    "nb_frames": "300",
                    "duration": "10.0",
                }
            ],
            "format": {"duration": "10.0"},
        }
        mock_probe.return_value = dummy_probe

        specs = get_video_specs("dummy.mp4")
        self.assertEqual(specs["frame_rate"], 30)
        self.assertEqual(specs["codec"], "h264")
        self.assertEqual(specs["width"], 640)
        self.assertEqual(specs["height"], 480)
        self.assertEqual(specs["nb_frames"], 300)
        self.assertEqual(specs["duration"], 10.0)

    @patch("ethology.video.utilities.ffmpeg.input")
    def test_extract_all_frames(self, mock_input):
        # Test extract_frames with all_frames=True.
        chain = MagicMock()
        chain.run.return_value = None
        mock_input.return_value.output.return_value = chain

        # Remove or create the output directory as needed
        test_output_dir = "test_frames_output"
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)

        extract_frames("input.mp4", test_output_dir, all_frames=True)
        mock_input.return_value.output.assert_called()
        chain.run.assert_called_once_with(quiet=True, overwrite_output=True)

    @patch("ethology.video.utilities.ffmpeg.input")
    @patch("ethology.video.utilities.get_video_specs")
    def test_extract_specific_frames(self, mock_get_specs, mock_input):
        # Setup dummy specs with 30 fps.
        mock_get_specs.return_value = {"frame_rate": 30}
        chain = MagicMock()
        chain.run.return_value = None
        mock_input.return_value.output.return_value = chain

        extract_frames("input.mp4", "output_dir", frame_numbers=[30])
        mock_input.assert_called_with(
            "input.mp4", ss=1.0
        )  # 30/30 = 1.0 sec timestamp
        chain.run.assert_called_once_with(quiet=True, overwrite_output=True)

    @patch("ethology.video.utilities.ffmpeg.input")
    def test_extract_time_range(self, mock_input):
        # Test extract_frames with a time range.
        chain = MagicMock()
        chain.run.return_value = None
        mock_input.return_value.output.return_value = chain

        extract_frames("input.mp4", "output_dir", time_range=(5, 10))
        # The call should include start=5 and t=5 (duration).
        mock_input.assert_called_with("input.mp4", ss=5, t=5)
        chain.run.assert_called_once_with(quiet=True, overwrite_output=True)

    @patch("ethology.video.utilities.ffmpeg.input")
    def test_combine_images_to_video(self, mock_input):
        chain = MagicMock()
        chain.run.return_value = None
        mock_input.return_value.output.return_value = chain

        combine_images_to_video(
            "frames/frame_%06d.jpg", "combined_video.mp4", fps=24
        )
        mock_input.assert_called_with("frames/frame_%06d.jpg", framerate=24)
        chain.run.assert_called_once_with(quiet=True, overwrite_output=True)

    @patch("ethology.video.utilities.ffmpeg.input")
    def test_crop_video_time(self, mock_input):
        chain = MagicMock()
        chain.run.return_value = None
        mock_input.return_value.output.return_value = chain

        crop_video_time("input.mp4", "cropped_time.mp4", start=10, end=20)
        mock_input.assert_called_with("input.mp4", ss=10, t=10)
        chain.run.assert_called_once_with(quiet=True, overwrite_output=True)

    @patch("ethology.video.utilities.ffmpeg.input")
    def test_crop_video_spatial(self, mock_input):
        chain = MagicMock()
        chain.run.return_value = None
        # We need to simulate the crop method in the chain.
        input_instance = MagicMock()
        input_instance.crop.return_value.output.return_value = chain
        mock_input.return_value = input_instance

        crop_video_spatial(
            "input.mp4",
            "cropped_spatial.mp4",
            x=100,
            y=50,
            width=640,
            height=360,
        )
        input_instance.crop.assert_called_with(100, 50, 640, 360)
        chain.run.assert_called_once_with(quiet=True, overwrite_output=True)

    @patch("ethology.video.utilities.ffmpeg.probe")
    def test_print_video_specs(self, mock_probe):
        # Since print_video_specs prints to terminal, we can capture the output.
        dummy_probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "r_frame_rate": "25/1",
                    "codec_name": "mpeg4",
                    "width": 800,
                    "height": 600,
                    "nb_frames": "250",
                    "duration": "10.0",
                }
            ],
            "format": {"duration": "10.0"},
        }
        mock_probe.return_value = dummy_probe

        import sys
        from io import StringIO

        captured_output = StringIO()
        sys.stdout = captured_output
        print_video_specs("dummy.mp4")
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn("Codec: mpeg4", output)
        self.assertIn("Frame Rate: 25.00 fps", output)
        self.assertIn("Duration: 10.0 seconds", output)
        self.assertIn("Number of Frames: 250", output)
        self.assertIn("Frame Size: 800x600", output)


if __name__ == "__main__":
    unittest.main()
