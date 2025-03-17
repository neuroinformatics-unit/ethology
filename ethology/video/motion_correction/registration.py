"""functions for motion correction and image registration."""

import itk
from parameters import get_default_parameters


def register_images(fixed_image, moving_image, parameter_object=None):
    """Perform rigid/affine registration between two images.

    Args:
        fixed_image: Reference image (ITK image object)
        moving_image: Image to align (ITK image object)
        parameter_object: Preconfigured parameters (Optional)

    Returns:
        registered_image: Aligned image
        transform_params: Resulting transformation parameters

    """
    if parameter_object is None:
        parameter_object = get_default_parameters()

    return itk.elastix_registration_method(
        fixed_image, moving_image, parameter_object=parameter_object
    )
