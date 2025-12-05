import torch

from models.inr.metasiren import MetaSiren, MetaReLU, ModularMetaSiren
from models.wrapper import MetaWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_inr(P):
    """
    Instantiates an implicit neural representation (INR) model based on the specified decoder type.

    Args:
        P (argparse.Namespace): Configuration object containing model hyperparameters such as:
            - P.decoder: Type of INR decoder ('siren' or 'relu')
            - P.dim_in: Input dimension
            - P.dim_hidden: Hidden layer width
            - P.dim_out: Output dimension
            - P.num_layers: Number of layers
            - P.w0: Frequency scaling parameter for SIREN
            - P.data_size, P.data_type, P.w0_type: Additional attributes for meta structure

    Returns:
        nn.Module: An instance of MetaSiren or MetaReLU
    """
    if P.decoder == "siren":
        model = MetaSiren(
            P.dim_in,
            P.dim_hidden,
            P.dim_out,
            P.num_layers,
            w0=P.w0,
            w0_initial=P.w0,
            data_size=P.data_size,
            data_type=P.data_type,
            w0_type=P.w0_type,
        )
    elif P.decoder == "mod_siren":
        model = ModularMetaSiren(
            P.dim_in,
            P.dim_hidden,
            P.dim_out,
            P.num_layers,
            w0=P.w0,
            w0_initial=P.w0,
            data_size=P.data_size,
            data_type=P.data_type,
            w0_type=P.w0_type,
            num_submodules=P.num_submodules,
            routing_order=P.order,
        )
    elif P.decoder == "relu":
        model = MetaReLU(
            P.dim_in,
            P.dim_hidden,
            P.dim_out,
            P.num_layers,
            w0=P.w0,
            w0_initial=P.w0,
            data_size=P.data_size,
            data_type=P.data_type,
            w0_type=P.w0_type,
        )
    else:
        raise ValueError("No such model exists. Please choose 'siren' or 'relu'.")

    return model


def get_model(P, centroids=None):
    """
    Wraps the selected INR decoder model in a MetaWrapper suitable for meta-learning.

    Args:
        P (argparse.Namespace): Configuration object containing model and data attributes.
            - Must include P.decoder and P.data_type (e.g., 'img', 'video')

    Returns:
        nn.Module: Wrapped meta-learnable model, suitable for training and adaptation.
    """
    decoder = get_inr(P)

    if P.data_type in ["img", "video", "ray"]:
        return MetaWrapper(P, decoder)
    else:

        raise NotImplementedError(
            "Data type not supported. Use 'img' or 'video' or 'ray'."
        )
