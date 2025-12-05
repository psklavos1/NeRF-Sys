def setup(algo, P):
    """
    Selects the appropriate evaluation function based on the training algo and data type.

    Args:
        algo (str): Training or evaluation algo (e.g., 'maml', 'fomaml', 'non-maml', etc.).
        P (argparse.Namespace): Configuration object containing data_type ('img' or 'video').

    Returns:
        function: Reference to the correct test function for model evaluation.
    """
    if algo in ["fomaml", "maml", "reptile"]:
        # Use standard MAML-style evaluation functions
        if P.data_type == "img":
            from evals.gradient_based.maml import test_model_img as test_func
        elif P.data_type == "video":
            from evals.gradient_based.maml import test_model_video as test_func
        elif P.data_type == "ray":
            from evals.gradient_based.nerf_eval import validate_nerf_model as test_func  
    else:
        # Fallback to default test_model with a warning
        print(f"Warning: current running option, i.e., {algo}, needs evaluation code")
        from evals.gradient_based.maml import test_model as test_func

    return test_func
