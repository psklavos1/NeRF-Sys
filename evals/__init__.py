def setup(algo, P):
    """
    Selects the appropriate evaluation function based on the training algo and data type.
    """
    if algo in ["fomaml", "maml"]:
        # Use standard MAML-style evaluation functions
        if P.data_type == "ray":
            from evals.maml import validate_nerf_model as test_func  
        else:
            raise NotImplementedError(f"MAML evaluation not implemented for data_type: {P.data_type}")
    return test_func
