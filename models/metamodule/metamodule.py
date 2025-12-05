from typing import Dict, Optional
import torch
import torch.nn as nn
import re
import warnings
from collections import OrderedDict
from einops import rearrange

from models.trunc_exp import trunc_exp


class MetaModule(nn.Module):
    """
    Base class for meta-learnable modules.

    Enables models to receive an optional `params` dictionary of parameters,
    allowing parameter substitution during the forward pass (e.g., for MAML, hypernetworks).
    """

    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix="", recurse=True):
        """
        Yields all meta-learnable parameters across nested MetaModules.
        """
        gen = self._named_members(
            lambda module: (
                module._parameters.items() if isinstance(module, MetaModule) else []
            ),
            prefix=prefix,
            recurse=recurse,
        )
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        """
        Returns an iterator over all meta-learnable parameters.
        """
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        """
        Retrieves the subset of `params` relevant to the submodule identified by `key`.

        Parameters:
        - params: Full parameter dictionary.
        - key: Submodule name (e.g., 'layer1').

        Returns:
        - An OrderedDict containing only the parameters associated with the submodule.
        """
        if params is None:
            return None

        all_names = tuple(params.keys())

        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names
            else:
                key_escape = re.escape(key)
                key_re = re.compile(rf"^{key_escape}\.(.+)")
                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r"\1", k) for k in all_names if key_re.match(k)
                ]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn(
                f"Module `{self.__class__.__name__}` has no parameter corresponding to submodule `{key}` in `params`.\n"
                f"Using default parameters. Provided keys: [{', '.join(all_names)}]",
                stacklevel=2,
            )
            return None

        return OrderedDict([(name, params[f"{key}.{name}"]) for name in names])


class MetaSequential(nn.Sequential, MetaModule):
    """
    A sequential container for meta-learnable layers.

    Works like `nn.Sequential`, but supports `params` injection per module.
    """

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=self.get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError(
                    f"The module must be a `nn.Module` or `MetaModule`. Got: {type(module)}"
                )
        return input


class MetaBatchLinear(nn.Linear, MetaModule):
    """
    Meta-learnable batched linear layer.

    This layer allows parameter substitution where weights and biases are
    defined per task (i.e., batched). It supports outer-loop optimization
    where each task may receive a different weight matrix.

    Parameters:
    - inputs: (batch_size, num_points, in_features)
    - weight: (batch_size, out_features, in_features)
    - bias:   (batch_size, out_features)
    """

    def forward(self, inputs, params=None):
        if params is None:
            # Default: use internal parameters, but expand them to match batch size
            params = OrderedDict(self.named_parameters())
            for name, param in params.items():
                params[name] = param[None, ...].repeat(
                    (inputs.size(0),) + (1,) * len(param.shape)
                )

        weight = params["weight"]  # shape: (B, out, in)
        bias = params.get("bias", None)  # shape: (B, out)

        # --- ensure batched shapes ---
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)  # (out,in) -> (1,out,in)
        if bias is not None:
            if bias.dim() == 1:
                bias = bias.unsqueeze(0)  # (out,)   -> (1,out)
            elif bias.dim() == 3 and bias.shape[1] == 1:  # (B,1,out) -> squeeze middle
                bias = bias.squeeze(1)

        # Reorder inputs to (B, in, N) for batched matmul
        inputs = rearrange(inputs, "b n d -> b d n")
        output = torch.bmm(weight, inputs)  # (B, out, N)
        output = rearrange(output, "b d n -> b n d")

        if bias is not None:
            output += bias.unsqueeze(1)  # (B, 1, out) for broadcasting over N

        return output


class MetaLinear(nn.Linear, MetaModule):
    """
    Minimal meta-learnable Linear (one task at a time).

    Forward:
      inputs: (num_points, in_features)
      params (optional): {
         "weight": (out_features, in_features),
         "bias":   (out_features,)  # optional
      }

    If params is None -> uses the module's own parameters.
    """

    def forward(
        self, inputs: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        # choose params
        if params is None:
            weight = self.weight
            bias = self.bias
        else:
            weight = params.get("weight", self.weight)
            bias = params.get("bias", self.bias)

        if inputs.dtype != weight.dtype:
            inputs = inputs.to(weight.dtype)

        # (..., in) x (in, out)^T -> (..., out)
        out = inputs.matmul(weight.t())
        if bias is not None:
            out = out + bias
        return out


# -------------------------------------------------
# MetaLinear + Activation wrapper (fast-weights ready)
# -------------------------------------------------
class MetaLayerBlock(MetaModule):
    def __init__(
        self, dim_in: int, dim_out: int, activation: Optional[str] = None, batched=False
    ):
        super().__init__()
        self.linear = (
            MetaLinear(dim_in, dim_out)
            if not batched
            else MetaBatchLinear(dim_in, dim_out)
        )
        if activation is None:
            self.act = nn.Identity()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation.lower() == "softplus":
            self.act = nn.Softplus()
        elif activation.lower() == "trunc_exp":
            self.act = trunc_exp
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x, params: Optional[OrderedDict] = None):
        x = self.linear(x, params=self.get_subdict(params, "linear"))
        return self.act(x)
