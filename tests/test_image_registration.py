import os
import unittest

from ethology.video.image_registration import register_images


class TestImageRegistration(unittest.TestCase):
    def setUp(self):
        self.fixed_image = "fixed.jpg"  # Replace with actual test images
        self.moving_image = "moving.jpg"
        self.output_image = "output.jpg"

    def test_registration(self):
        register_images(self.fixed_image, self.moving_image, self.output_image)
        self.assertTrue(os.path.exists(self.output_image))


if __name__ == "__main__":
    unittest.main()
