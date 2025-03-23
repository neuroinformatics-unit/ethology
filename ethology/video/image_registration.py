"""
This module provides image registration functionality using Elastix (SimpleITK).
"""
import SimpleITK as sitk


def register_images(fixed_image_path, moving_image_path, output_path):
    """
    Register a moving image to a fixed image using Elastix.

    This function computes a rigid transformation to align 
    the moving image with the fixed image.

    :param fixed_image_path: Path to the reference image.
    :param moving_image_path: Path to the image to be aligned.
    :param output_path: Path to save the registered image.
    """ 
    # Read images using SimpleITK
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Setup Elastix registration
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed_image)
    elastix.SetMovingImage(moving_image)

    # Load default parameter settings
    parameter_map = sitk.GetDefaultParameterMap("rigid")
    elastix.SetParameterMap(parameter_map)

    # Run registration
    elastix.Execute()

    # Get the registered image
    result_image = elastix.GetResultImage()

    # Save the registered image
    sitk.WriteImage(result_image, output_path)

    print(f"Registered image saved to: {output_path}")
