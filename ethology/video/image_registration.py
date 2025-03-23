"""Provides image registration functionality using Elastix (SimpleITK).
This module helps align a moving image to a fixed image using a rigid transform.
"""

import SimpleITK as sitk


def register_images(fixed_image_path, moving_image_path, output_path):
    """Register a moving image to a fixed image using Elastix.
    :param fixed_image_path: Path to the reference image.
    :param moving_image_path: Path to the image to be aligned.
    :param output_path: Path to save the registered image.
    """
    # Read images using SimpleITK
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Set up Elastix image filter
    elastix_filter = sitk.ElastixImageFilter()
    elastix_filter.SetFixedImage(fixed_image)
    elastix_filter.SetMovingImage(moving_image)

    # Apply registration
    elastix_filter.Execute()

    # Save registered image
    registered_image = elastix_filter.GetResultImage()
    sitk.WriteImage(registered_image, output_path)

    print(f"Registered image saved to: {output_path}")
