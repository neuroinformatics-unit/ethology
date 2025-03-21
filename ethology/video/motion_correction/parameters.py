"""default parameters for motion correction."""

import itk


def get_default_rigid_params():
    """Optimal parameters for drone footage translation."""
    params = {
        "Transform": ["Translation"],
        "NumberOfResolutions": ["4"],
        "MaximumNumberOfIterations": ["1000"],
        "Metric": ["AdvancedMattesMutualInformation"],
    }
    return create_parameter_object(params)


def create_parameter_object(param_dict):
    """Convert dictionary to ITK parameter object."""
    param_obj = itk.ParameterObject.New()
    param_map = param_obj.GetDefaultParameterMap("rigid")

    for key, value in param_dict.items():
        param_map[key] = value

    param_obj.AddParameterMap(param_map)
    return param_obj
