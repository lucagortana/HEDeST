from __future__ import annotations

import collections
import gc
import logging
from collections.abc import Iterable
from copy import deepcopy
from datetime import date
from functools import partial
from typing import Any
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import scanpy
import torch
from anndata import AnnData
from lightning.pytorch.callbacks import Callback
from pyro import clear_param_store
from pyro import poutine
from pyro.distributions.distribution import Distribution
from pyro.distributions.transforms import SoftplusTransform
from pyro.infer import Trace_ELBO
from pyro.infer import TraceEnum_ELBO
from pyro.infer.autoguide import AutoHierarchicalNormalMessenger
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.initialization import init_to_feasible
from pyro.infer.autoguide.initialization import init_to_mean
from pyro.infer.autoguide.utils import deep_getattr
from pyro.infer.autoguide.utils import deep_setattr
from pyro.infer.autoguide.utils import helpful_support_errors
from pyro.nn.module import PyroModule
from pyro.nn.module import PyroParam
from pyro.nn.module import to_pyro_module_
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalJointObsField
from scvi.data.fields import CategoricalObsField
from scvi.data.fields import LayerField
from scvi.data.fields import NumericalJointObsField
from scvi.data.fields import NumericalObsField
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import parse_device_args
from scvi.model.base import BaseModelClass
from scvi.model.base import PyroSampleMixin
from scvi.model.base import PyroSviTrainMixin
from scvi.model.base._pyromixin import PyroJitGuideWarmup
from scvi.module.base import PyroBaseModuleClass
from scvi.train import PyroTrainingPlan as PyroTrainingPlan_scvi
from scvi.train import TrainRunner
from scvi.utils import setup_anndata_dsp
from torch import nn as nn
from torch.distributions import biject_to
from torch.distributions import constraints
from torch.nn.functional import one_hot


def filter_genes(adata, cell_count_cutoff=15, cell_percentage_cutoff2=0.05, nonz_mean_cutoff=1.12, plot=False):
    r"""Plot the gene filter given a set of cutoffs and return resulting list of genes.

    Parameters
    ----------
    adata :
        anndata object with single cell / nucleus data.
    cell_count_cutoff :
        All genes detected in less than cell_count_cutoff cells will be excluded.
    cell_percentage_cutoff2 :
        All genes detected in at least this percentage of cells will be included.
    nonz_mean_cutoff :
        genes detected in the number of cells between the above mentioned cutoffs are selected
        only when their average expression in non-zero cells is above this cutoff.

    Returns
    -------
    a list of selected var_names
    """

    adata.var["n_cells"] = np.array((adata.X > 0).sum(0)).flatten()
    adata.var["nonz_mean"] = np.array(adata.X.sum(0)).flatten() / adata.var["n_cells"]

    cell_count_cutoff = np.log10(cell_count_cutoff)
    cell_count_cutoff2 = np.log10(adata.shape[0] * cell_percentage_cutoff2)
    nonz_mean_cutoff = np.log10(nonz_mean_cutoff)

    gene_selection = (np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff2)) | (
        np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff)
        & np.array(np.log10(adata.var["nonz_mean"]) > nonz_mean_cutoff)
    )
    gene_selection = adata.var_names[gene_selection]
    adata_shape = adata[:, gene_selection].shape

    if plot:
        fig, ax = plt.subplots()
        ax.hist2d(
            np.log10(adata.var["nonz_mean"]),
            np.log10(adata.var["n_cells"]),
            bins=100,
            norm=matplotlib.colors.LogNorm(),
            range=[[0, 0.5], [1, 4.5]],
        )
        ax.axvspan(0, nonz_mean_cutoff, ymin=0.0, ymax=(cell_count_cutoff2 - 1) / 3.5, color="darkorange", alpha=0.3)
        ax.axvspan(
            nonz_mean_cutoff,
            np.max(np.log10(adata.var["nonz_mean"])),
            ymin=0.0,
            ymax=(cell_count_cutoff - 1) / 3.5,
            color="darkorange",
            alpha=0.3,
        )
        plt.vlines(nonz_mean_cutoff, cell_count_cutoff, cell_count_cutoff2, color="darkorange")
        plt.hlines(cell_count_cutoff, nonz_mean_cutoff, 1, color="darkorange")
        plt.hlines(cell_count_cutoff2, 0, nonz_mean_cutoff, color="darkorange")
        plt.xlabel("Mean non-zero expression level of gene (log)")
        plt.ylabel("Number of cells expressing gene (log)")
        plt.title(f"Gene filter: {adata_shape[0]} cells x {adata_shape[1]} genes")
        plt.show()

    return gene_selection


class FCLayers(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.
    Dropout is performed on input rather than on output.

    Adapted with modifications from scvi-tools:
    Copyright (c) 2020 Romain Lopez, Adam Gayoso, Galen Xing, Yosef Lab
    All rights reserved.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                            nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                            activation_fn() if use_activation else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1))) for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


class FCLayersPyro(FCLayers, PyroModule):
    pass


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


class AutoAmortisedHierarchicalNormalMessenger(AutoHierarchicalNormalMessenger):
    """
    EXPERIMENTAL Automatic :class:`~pyro.infer.effect_elbo.GuideMessenger` ,
    intended for use with :class:`~pyro.infer.effect_elbo.Effect_ELBO` or
    similar. Amortise specific sites

    The mean-field posterior at any site is a transformed normal distribution,
    the mean of which depends on the value of that site given its dependencies in the model:

        loc = loc + transform.inv(prior.mean) * weight

    Where the value of `prior.mean` is conditional on upstream sites in the model.
    This approach doesn't work for distributions that don't have the mean.

    loc, scales and element-specific weight are amortised for each site specified in `amortised_plate_sites`.

    Derived classes may override particular sites and use this simply as a
    default, see AutoNormalMessenger documentation for example.

    :param callable model: A Pyro model.
    :param dict amortised_plate_sites: Dictionary with amortised plate details:
        the name of observation/minibatch plate,
        indexes of model args to provide to encoder,
        variable names that belong to the observation plate
        and the number of dimensions in non-plate axis of each variable - such as:
        {
            "name": "obs_plate",
            "input": [0],  # expression data + (optional) batch index ([0, 2])
            "input_transform": [torch.log1p], # how to transform input data before passing to NN
            "sites": {
                "n_s": 1,
                "y_s": 1,
                "z_sr": R,
                "w_sf": F,
            }
        }
    :param int n_in: Number of input dimensions (for encoder_class).
    :param int n_hidden: Number of hidden nodes in each layer, one of 3 options:
        1. Integer denoting the number of hidden nodes
        2. Dictionary with {"single": 200, "multiple": 200} denoting the number of hidden nodes for each `encoder_mode` (See below)
        3. Allowing different number of hidden nodes for each model site. Dictionary with the number of hidden nodes for single encode mode and each model site:
        {
            "single": 200
            "n_s": 5,
            "y_s": 5,
            "z_sr": 128,
            "w_sf": 200,
        }
    :param float init_param_scale: How to scale/normalise initial values for weights converting hidden layers to loc and scales.
    :param float scales_offset: offset between the output of the NN and scales.
    :param Callable encoder_class: Class that defines encoder network.
    :param dict encoder_kwargs: Keyword arguments for encoder class.
    :param dict multi_encoder_kwargs: Optional separate keyword arguments for encoder_class,
        useful when encoder_mode == "single-multiple".
    :param Callable encoder_instance: Encoder network instance, overrides class input and the input instance is copied with deepcopy.
    :param str encoder_mode: Use single encoder for all variables ("single"), one encoder per variable ("multiple")
        or a single encoder in the first step and multiple encoders in the second step ("single-multiple").
    :param list hierarchical_sites: List of latent variables (model sites)
        that have hierarchical dependencies.
        If None, all sites are assumed to have hierarchical dependencies. If None, for the sites
        that don't have upstream sites, the guide is representing/learning deviation from the prior.
    """

    # 'element-wise' or 'scalar'
    weight_type = "element-wise"

    def __init__(
        self,
        model: Callable,
        *,
        amortised_plate_sites: dict,
        n_in: int,
        n_hidden: dict = None,
        init_param_scale: float = 1 / 50,
        init_scale: float = 0.1,
        init_weight: float = 1.0,
        init_loc_fn: Callable = init_to_mean(fallback=init_to_feasible),
        encoder_class=FCLayersPyro,
        encoder_kwargs=None,
        multi_encoder_kwargs=None,
        encoder_instance: torch.nn.Module = None,
        encoder_mode: Literal["single", "multiple", "single-multiple"] = "single",
        hierarchical_sites: Optional[list] = None,
        bias=True,
        use_posterior_lsw_encoders=False,
    ):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        super().__init__(model, init_loc_fn=init_loc_fn)
        self._init_scale = init_scale
        self._init_weight = init_weight
        self._hierarchical_sites = hierarchical_sites
        self.amortised_plate_sites = amortised_plate_sites
        self.encoder_mode = encoder_mode
        self.bias = bias
        self.use_posterior_lsw_encoders = use_posterior_lsw_encoders
        self._computing_median = False
        self._computing_quantiles = False
        self._quantile_values = None
        self._computing_mi = False
        self.mi = dict()
        self.samples_for_mi = None

        self.softplus = SoftplusTransform()

        # default n_hidden values and checking input
        if n_hidden is None:
            n_hidden = {"single": 200, "multiple": 200}
        else:
            if isinstance(n_hidden, int):
                n_hidden = {"single": n_hidden, "multiple": n_hidden}
            elif not isinstance(n_hidden, dict):
                raise ValueError("n_hidden must be either int or dict")
        # process encoder kwargs, add n_hidden, create argument for multiple encoders
        encoder_kwargs = deepcopy(encoder_kwargs) if isinstance(encoder_kwargs, dict) else dict()
        encoder_kwargs["n_hidden"] = n_hidden["single"]
        if multi_encoder_kwargs is None:
            multi_encoder_kwargs = deepcopy(encoder_kwargs)

        # save encoder parameters
        self.encoder_kwargs = encoder_kwargs
        self.multi_encoder_kwargs = multi_encoder_kwargs
        self.single_n_in = n_in
        self.multiple_n_in = n_in
        self.n_hidden = n_hidden
        if ("single" in encoder_mode) and ("multiple" in encoder_mode):
            # if single network precedes multiple networks
            self.multiple_n_in = self.n_hidden["single"]
        self.encoder_class = encoder_class
        self.encoder_instance = encoder_instance
        self.init_param_scale = init_param_scale

    def get_posterior(
        self,
        name: str,
        prior: Distribution,
    ) -> Union[Distribution, torch.Tensor]:
        if self._computing_median:
            return self._get_posterior_median(name, prior)
        if self._computing_quantiles:
            return self._get_posterior_quantiles(name, prior)
        if self._computing_mi:
            # the messenger autoguide needs the output to fit certain dimensions
            # this is hack which saves MI to self.mi but returns cheap to compute medians
            self.mi[name] = self._get_mutual_information(name, prior)
            return self._get_posterior_median(name, prior)

        with helpful_support_errors({"name": name, "fn": prior}):
            transform = biject_to(prior.support)
        # If hierarchical_sites not specified all sites are assumed to be hierarchical
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight  # - torch.tensor(3.0, device=prior.mean.device)
            posterior = dist.TransformedDistribution(
                dist.Normal(loc, scale).to_event(transform.domain.event_dim),
                transform.with_cache(),
            )
            return posterior
        else:
            # Fall back to mean field when hierarchical_sites list is not empty and site not in the list.
            loc, scale = self._get_params(name, prior)
            posterior = dist.TransformedDistribution(
                dist.Normal(loc, scale).to_event(transform.domain.event_dim),
                transform.with_cache(),
            )
            return posterior

    def encode(self, name: str, prior: Distribution):
        """
        Apply encoder network to input data to obtain hidden layer encoding.
        Parameters
        ----------
        args
            Pyro model args
        kwargs
            Pyro model kwargs
        -------

        """
        try:
            args, kwargs = self.args_kwargs  # stored as a tuple of (tuple, dict)
            # get the data for NN from
            in_names = self.amortised_plate_sites["input"]
            x_in = [kwargs[i] if i in kwargs.keys() else args[i] for i in in_names]
            # apply data transform before passing to NN
            site_transform = self.amortised_plate_sites.get("site_transform", None)
            if site_transform is not None and name in site_transform.keys():
                # when input data transform and input dimensions differ between variables
                in_transforms = site_transform[name]["input_transform"]
                single_n_in = site_transform[name]["n_in"]
                multiple_n_in = site_transform[name]["n_in"]
                if ("single" in self.encoder_mode) and ("multiple" in self.encoder_mode):
                    # if single network precedes multiple networks
                    multiple_n_in = self.multiple_n_in
            else:
                in_transforms = self.amortised_plate_sites["input_transform"]
                single_n_in = self.single_n_in
                multiple_n_in = self.multiple_n_in

            x_in = [in_transforms[i](x) for i, x in enumerate(x_in)]
            # apply learnable normalisation before passing to NN:
            input_normalisation = self.amortised_plate_sites.get("input_normalisation", None)
            if input_normalisation is not None:
                for i in range(len(self.amortised_plate_sites["input"])):
                    if input_normalisation[i]:
                        x_in[i] = x_in[i] * deep_getattr(self, f"input_normalisation_{i}")
            if "single" in self.encoder_mode:
                # encode with a single encoder
                res = deep_getattr(self, "one_encoder")(*x_in)
                if "multiple" in self.encoder_mode:
                    # when there is a second layer of multiple encoders fetch encoders and encode data
                    x_in[0] = res
                    res = deep_getattr(self.multiple_encoders, name)(*x_in)
            else:
                # when there are multiple encoders fetch encoders and encode data
                res = deep_getattr(self.multiple_encoders, name)(*x_in)
            return res
        except AttributeError:
            pass

        # Initialize.
        # create normalisation parameters if necessary:
        input_normalisation = self.amortised_plate_sites.get("input_normalisation", None)
        if input_normalisation is not None:
            for i in range(len(self.amortised_plate_sites["input"])):
                if input_normalisation[i]:
                    deep_setattr(
                        self,
                        f"input_normalisation_{i}",
                        PyroParam(torch.ones((1, single_n_in)).to(prior.mean.device).requires_grad_(True)),
                    )
        # create encoder neural networks
        if "single" in self.encoder_mode:
            if self.encoder_instance is not None:
                # copy provided encoder instance
                one_encoder = deepcopy(self.encoder_instance).to(prior.mean.device)
                # convert to pyro module
                to_pyro_module_(one_encoder)
                deep_setattr(self, "one_encoder", one_encoder)
            else:
                # create encoder instance from encoder class
                deep_setattr(
                    self,
                    "one_encoder",
                    self.encoder_class(n_in=single_n_in, n_out=self.n_hidden["single"], **self.encoder_kwargs).to(
                        prior.mean.device
                    ),
                )
        if "multiple" in self.encoder_mode:
            # determine the number of hidden layers
            if name in self.n_hidden.keys():
                n_hidden = self.n_hidden[name]
            else:
                n_hidden = self.n_hidden["multiple"]
            multi_encoder_kwargs = deepcopy(self.multi_encoder_kwargs)
            multi_encoder_kwargs["n_hidden"] = n_hidden

            # create multiple encoders
            if self.encoder_instance is not None:
                # copy instances
                encoder_ = deepcopy(self.encoder_instance).to(prior.mean.device)
                # convert to pyro module
                to_pyro_module_(encoder_)
                deep_setattr(
                    self,
                    "multiple_encoders." + name,
                    encoder_,
                )
            else:
                # create instances
                deep_setattr(
                    self,
                    "multiple_encoders." + name,
                    self.encoder_class(n_in=multiple_n_in, n_out=n_hidden, **multi_encoder_kwargs).to(
                        prior.mean.device
                    ),
                )
        return self.encode(name, prior)

    def _get_params(self, name: str, prior: Distribution):
        if name not in self.amortised_plate_sites["sites"].keys():
            # don't use amortisation unless requested (site in the list)
            return super()._get_params(name, prior)

        args, kwargs = self.args_kwargs  # stored as a tuple of (tuple, dict)
        hidden = self.encode(name, prior)
        try:
            linear_loc = deep_getattr(self.hidden2locs, name)
            linear_scale = deep_getattr(self.hidden2scales, name)
            if not self.use_posterior_lsw_encoders:
                loc = linear_loc(hidden)
                scale = self.softplus(linear_scale(hidden) + self._init_scale_unconstrained)
            else:
                args, kwargs = self.args_kwargs  # stored as a tuple of (tuple, dict)
                # get the data for NN from
                in_names = self.amortised_plate_sites["input"]
                x_in = [kwargs[i] if i in kwargs.keys() else args[i] for i in in_names]
                x_in[0] = hidden
                # apply data transform before passing to NN
                site_transform = self.amortised_plate_sites.get("site_transform", None)
                if site_transform is not None and name in site_transform.keys():
                    # when input data transform and input dimensions differ between variables
                    in_transforms = site_transform[name]["input_transform"]
                else:
                    in_transforms = self.amortised_plate_sites["input_transform"]
                x_in = [in_transforms[i](x) if i != 0 else x for i, x in enumerate(x_in)]
                linear_loc_encoder = deep_getattr(self.hidden2locs, f"{name}.encoder")
                linear_scale_encoder = deep_getattr(self.hidden2scales, f"{name}.encoder")
                loc = linear_loc(linear_loc_encoder(*x_in))
                scale = self.softplus(linear_scale(linear_scale_encoder(*x_in)) + self._init_scale_unconstrained)
            if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                if self.weight_type == "element-wise":
                    # weight is element-wise
                    linear_weight = deep_getattr(self.hidden2weights, name)
                    if not self.use_posterior_lsw_encoders:
                        weight = self.softplus(linear_weight(hidden) + self._init_weight_unconstrained)
                    else:
                        linear_weight_encoder = deep_getattr(self.hidden2weights, f"{name}.encoder")
                        weight = self.softplus(
                            linear_weight(linear_weight_encoder(hidden)) + self._init_weight_unconstrained
                        )
                if self.weight_type == "scalar":
                    # weight is a single value parameter
                    weight = deep_getattr(self.weights, name)
                return loc, scale, weight
            else:
                return loc, scale
        except AttributeError:
            pass

        # Initialize.
        with torch.no_grad():
            init_scale = torch.full((), self._init_scale)
            self._init_scale_unconstrained = self.softplus.inv(init_scale)
            init_weight = torch.full((), self._init_weight)
            self._init_weight_unconstrained = self.softplus.inv(init_weight)

            # determine the number of hidden layers
            if "multiple" in self.encoder_mode:
                if name in self.n_hidden.keys():
                    n_hidden = self.n_hidden[name]
                else:
                    n_hidden = self.n_hidden["multiple"]
            elif "single" in self.encoder_mode:
                n_hidden = self.n_hidden["single"]
            # determine parameter dimensions
            out_dim = self.amortised_plate_sites["sites"][name]

        deep_setattr(
            self,
            "hidden2locs." + name,
            PyroModule[torch.nn.Linear](n_hidden, out_dim, bias=self.bias, device=prior.mean.device),
        )
        deep_setattr(
            self,
            "hidden2scales." + name,
            PyroModule[torch.nn.Linear](n_hidden, out_dim, bias=self.bias, device=prior.mean.device),
        )

        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            if self.weight_type == "scalar":
                # weight is a single value parameter
                deep_setattr(self, "weights." + name, PyroParam(init_weight, constraint=constraints.positive))
            if self.weight_type == "element-wise":
                # weight is element-wise
                deep_setattr(
                    self,
                    "hidden2weights." + name,
                    PyroModule[torch.nn.Linear](n_hidden, out_dim, bias=self.bias, device=prior.mean.device),
                )

        if self.use_posterior_lsw_encoders:
            # determine the number of hidden layers
            if name in self.n_hidden.keys():
                n_hidden = self.n_hidden[name]
            else:
                n_hidden = self.n_hidden["multiple"]
            multi_encoder_kwargs = deepcopy(self.multi_encoder_kwargs)
            multi_encoder_kwargs["n_hidden"] = n_hidden

            # create multiple encoders
            if self.encoder_instance is not None:
                # copy instances
                encoder_ = deepcopy(self.encoder_instance).to(prior.mean.device)
                # convert to pyro module
                to_pyro_module_(encoder_)
                deep_setattr(
                    self,
                    f"hidden2locs.{name}.encoder",
                    encoder_,
                )
                deep_setattr(
                    self,
                    f"hidden2scales.{name}.encoder",
                    encoder_,
                )
                if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                    deep_setattr(
                        self,
                        f"hidden2weights.{name}.encoder",
                        encoder_,
                    )
            else:
                # create instances
                deep_setattr(
                    self,
                    f"hidden2locs.{name}.encoder",
                    self.encoder_class(n_in=n_hidden, n_out=n_hidden, **multi_encoder_kwargs).to(prior.mean.device),
                )
                deep_setattr(
                    self,
                    f"hidden2scales.{name}.encoder",
                    self.encoder_class(n_in=n_hidden, n_out=n_hidden, **multi_encoder_kwargs).to(prior.mean.device),
                )
                if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                    deep_setattr(
                        self,
                        f"hidden2weights.{name}.encoder",
                        self.encoder_class(n_in=n_hidden, n_out=n_hidden, **multi_encoder_kwargs).to(prior.mean.device),
                    )

        return self._get_params(name, prior)

    def median(self, *args, **kwargs):
        self._computing_median = True
        try:
            return self(*args, **kwargs)
        finally:
            self._computing_median = False

    @torch.no_grad()
    def _get_posterior_median(self, name, prior):
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)
        return transform(loc)

    def quantiles(self, quantiles, *args, **kwargs):
        self._computing_quantiles = True
        self._quantile_values = quantiles
        try:
            return self(*args, **kwargs)
        finally:
            self._computing_quantiles = False

    @torch.no_grad()
    def _get_posterior_quantiles(self, name, prior):
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)

        site_quantiles = torch.tensor(self._quantile_values, dtype=loc.dtype, device=loc.device)
        site_quantiles_values = dist.Normal(loc, scale).icdf(site_quantiles)
        return transform(site_quantiles_values)

    def mutual_information(self, *args, **kwargs):
        # compute samples necessary to compute MI
        self.samples_for_mi = self(*args, **kwargs)
        self._computing_mi = True
        try:
            # compute mi (saved to self.mi)
            self(*args, **kwargs)
            return self.mi
        finally:
            self._computing_mi = False

    @torch.no_grad()
    def _get_mutual_information(self, name, prior):
        """Approximate the mutual information between data x and latent variable z

            I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Returns: Float

        """

        #### get posterior mean and variance ####
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)

        if name not in self.amortised_plate_sites["sites"].keys():
            # if amortisation is not used for a particular site return MI=0
            return 0

        #### create tensors with useful numbers ####
        one = torch.ones((), dtype=loc.dtype, device=loc.device)
        two = torch.tensor(2, dtype=loc.dtype, device=loc.device)
        pi = torch.tensor(3.14159265359, dtype=loc.dtype, device=loc.device)
        #### get sample from posterior ####
        z_samples = self.samples_for_mi[name]

        #### compute mi ####
        x_batch, nz = loc.size()
        x_batch = torch.tensor(x_batch, dtype=loc.dtype, device=loc.device)
        nz = torch.tensor(nz, dtype=loc.dtype, device=loc.device)

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+scale.loc()).sum(-1)
        neg_entropy = (
            -nz * torch.log(pi * two) * (one / two) - ((scale**two).log() + one).sum(-1) * (one / two)
        ).mean()

        # [1, x_batch, nz]
        loc, scale = loc.unsqueeze(0), scale.unsqueeze(0)
        var = scale**two

        # (z_batch, x_batch, nz)
        dev = z_samples - loc

        # (z_batch, x_batch)
        log_density = -((dev**two) / var).sum(dim=-1) * (one / two) - (
            nz * torch.log(pi * two) + (scale**two).log().sum(-1)
        ) * (one / two)

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - torch.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()


class QuantileMixin:
    """
    This mixin class provides methods for:

    - computing median and quantiles of the posterior distribution using both direct and amortised inference

    """

    def _optim_param(
        self,
        lr: float = 0.01,
        autoencoding_lr: float = None,
        clip_norm: float = 200,
        module_names: list = ["encoder", "hidden2locs", "hidden2scales"],
    ):
        # TODO implement custom training method that can use this function.
        # create function which fetches different lr for autoencoding guide
        def optim_param(module_name, param_name):
            # detect variables in autoencoding guide
            if autoencoding_lr is not None and np.any([n in module_name + "." + param_name for n in module_names]):
                return {
                    "lr": autoencoding_lr,
                    # limit the gradient step from becoming too large
                    "clip_norm": clip_norm,
                }
            else:
                return {
                    "lr": lr,
                    # limit the gradient step from becoming too large
                    "clip_norm": clip_norm,
                }

        return optim_param

    @torch.no_grad()
    def _posterior_quantile_minibatch(
        self,
        q: float = 0.5,
        batch_size: int = 2048,
        use_gpu: bool = None,
        use_median: bool = True,
        exclude_vars: list = None,
        data_loader_indices=None,
    ):
        """
        Compute median of the posterior distribution of each parameter, separating local (minibatch) variable
        and global variables, which is necessary when performing amortised inference.

        Note for developers: requires model class method which lists observation/minibatch plate
        variables (self.module.model.list_obs_plate_vars()).

        Parameters
        ----------
        q
            quantile to compute
        batch_size
            number of observations per batch
        use_gpu
            Bool, use gpu?
        use_median
            Bool, when q=0.5 use median rather than quantile method of the guide

        Returns
        -------
        dictionary {variable_name: posterior quantile}

        """

        _, _, device = parse_device_args(use_gpu, return_device="torch")

        self.module.eval()

        train_dl = AnnDataLoader(self.adata_manager, shuffle=False, batch_size=batch_size, indices=data_loader_indices)

        # sample local parameters
        i = 0
        for tensor_dict in train_dl:

            args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
            args = [a.to(device) for a in args]
            kwargs = {k: v.to(device) for k, v in kwargs.items()}
            self.to_device(device)

            if i == 0:
                # find plate sites
                obs_plate_sites = self._get_obs_plate_sites(args, kwargs, return_observed=True)
                if len(obs_plate_sites) == 0:
                    # if no local variables - don't sample
                    break
                # find plate dimension
                obs_plate_dim = list(obs_plate_sites.values())[0]
                if use_median and q == 0.5:
                    means = self.module.guide.median(*args, **kwargs)
                else:
                    means = self.module.guide.quantiles([q], *args, **kwargs)
                means = {
                    k: means[k].cpu().numpy()
                    for k in means.keys()
                    if (k in obs_plate_sites) and (k not in exclude_vars)
                }

            else:
                if use_median and q == 0.5:
                    means_ = self.module.guide.median(*args, **kwargs)
                else:
                    means_ = self.module.guide.quantiles([q], *args, **kwargs)
                means_ = {
                    k: means_[k].cpu().numpy()
                    for k in means_.keys()
                    if (k in obs_plate_sites) and (k not in exclude_vars)
                }
                means = {k: np.concatenate([means[k], means_[k]], axis=obs_plate_dim) for k in means.keys()}
            i += 1

        # sample global parameters
        tensor_dict = next(iter(train_dl))
        args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
        args = [a.to(device) for a in args]
        kwargs = {k: v.to(device) for k, v in kwargs.items()}
        self.to_device(device)

        if use_median and q == 0.5:
            global_means = self.module.guide.median(*args, **kwargs)
        else:
            global_means = self.module.guide.quantiles([q], *args, **kwargs)
        global_means = {
            k: global_means[k].cpu().numpy()
            for k in global_means.keys()
            if (k not in obs_plate_sites) and (k not in exclude_vars)
        }

        for k in global_means.keys():
            means[k] = global_means[k]

        # quantile returns tensors with 0th dimension = 1
        if not (use_median and q == 0.5) and (
            not isinstance(self.module.guide, AutoAmortisedHierarchicalNormalMessenger)
        ):
            means = {k: means[k].squeeze(0) for k in means.keys()}

        self.module.to(device)

        return means

    @torch.no_grad()
    def _posterior_quantile(
        self,
        q: float = 0.5,
        batch_size: int = None,
        use_gpu: bool = None,
        use_median: bool = True,
        exclude_vars: list = None,
        data_loader_indices=None,
    ):
        """
        Compute median of the posterior distribution of each parameter pyro models trained without amortised inference.

        Parameters
        ----------
        q
            Quantile to compute
        use_gpu
            Bool, use gpu?
        use_median
            Bool, when q=0.5 use median rather than quantile method of the guide

        Returns
        -------
        dictionary {variable_name: posterior quantile}

        """

        self.module.eval()
        _, _, device = parse_device_args(use_gpu, return_device="torch")
        if batch_size is None:
            batch_size = self.adata_manager.adata.n_obs
        train_dl = AnnDataLoader(self.adata_manager, shuffle=False, batch_size=batch_size, indices=data_loader_indices)
        # sample global parameters
        tensor_dict = next(iter(train_dl))
        args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
        args = [a.to(device) for a in args]
        kwargs = {k: v.to(device) for k, v in kwargs.items()}
        self.to_device(device)

        if use_median and q == 0.5:
            means = self.module.guide.median(*args, **kwargs)
        else:
            means = self.module.guide.quantiles([q], *args, **kwargs)
        means = {k: means[k].cpu().detach().numpy() for k in means.keys() if k not in exclude_vars}

        # quantile returns tensors with 0th dimension = 1
        if not (use_median and q == 0.5) and (
            not isinstance(self.module.guide, AutoAmortisedHierarchicalNormalMessenger)
        ):
            means = {k: means[k].squeeze(0) for k in means.keys()}

        return means

    def posterior_quantile(self, exclude_vars: list = None, batch_size: int = None, **kwargs):
        """
        Compute median of the posterior distribution of each parameter.

        Parameters
        ----------
        q
            Quantile to compute
        use_gpu
            Bool, use gpu?
        use_median
            Bool, when q=0.5 use median rather than quantile method of the guide

        Returns
        -------

        """
        if exclude_vars is None:
            exclude_vars = []
        if kwargs is None:
            kwargs = dict()

        if isinstance(self.module.guide, AutoNormal):
            # median/quantiles in AutoNormal does not require minibatches
            batch_size = None

        if batch_size is not None:
            return self._posterior_quantile_minibatch(exclude_vars=exclude_vars, batch_size=batch_size, **kwargs)
        else:
            return self._posterior_quantile(exclude_vars=exclude_vars, batch_size=batch_size, **kwargs)


class PltExportMixin:
    r"""
    This mixing class provides methods for common plotting tasks and data export.
    """

    @staticmethod
    def plot_posterior_mu_vs_data(mu, data):
        r"""Plot expected value of the model (e.g. mean of NB distribution) vs observed data

        :param mu: expected value
        :param data: data value
        """

        plt.hist2d(
            np.log10(data.flatten() + 1),
            np.log10(mu.flatten() + 1),
            bins=50,
            norm=matplotlib.colors.LogNorm(),
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("Data, log10")
        plt.ylabel("Posterior expected value, log10")
        plt.title("Reconstruction accuracy")
        plt.tight_layout()

    def plot_history(self, iter_start=0, iter_end=-1, ax=None):
        r"""Plot training history
        Parameters
        ----------
        iter_start
            omit initial iterations from the plot
        iter_end
            omit last iterations from the plot
        ax
            matplotlib axis
        """
        if ax is None:
            ax = plt.gca()
        if iter_end == -1:
            iter_end = len(self.history_["elbo_train"])

        ax.plot(
            np.array(self.history_["elbo_train"].index[iter_start:iter_end]),
            np.array(self.history_["elbo_train"].values.flatten())[iter_start:iter_end],
            label="train",
        )
        ax.legend()
        ax.set_xlim(0, len(self.history_["elbo_train"]))
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("-ELBO loss")
        plt.tight_layout()

    def _export2adata(self, samples):
        r"""
        Export key model variables and samples

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``

        Returns
        -------
            Updated dictionary with additional details is saved to ``adata.uns['mod']``.
        """
        # add factor filter and samples of all parameters to unstructured data
        results = {
            "model_name": str(self.module.__class__.__name__),
            "date": str(date.today()),
            "factor_filter": list(getattr(self, "factor_filter", [])),
            "factor_names": list(self.factor_names_),
            "var_names": self.adata.var_names.tolist(),
            "obs_names": self.adata.obs_names.tolist(),
            "post_sample_means": samples["post_sample_means"] if "post_sample_means" in samples else None,
            "post_sample_stds": samples["post_sample_stds"] if "post_sample_stds" in samples else None,
        }
        # add posterior quantiles
        for k, v in samples.items():
            if k.startswith("post_sample_"):
                results[k] = v
        if type(self.factor_names_) is dict:
            results["factor_names"] = self.factor_names_

        return results

    def sample2df_obs(
        self,
        samples: dict,
        site_name: str = "w_sf",
        summary_name: str = "means",
        name_prefix: str = "cell_abundance",
        factor_names_key: str = "",
    ):
        """Export posterior distribution summary for observation-specific parameters
        (e.g. spatial cell abundance) as Pandas data frame
        (means, 5%/95% quantiles or sd of posterior distribution).

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``
        site_name
            name of the model parameter to be exported
        summary_name
            posterior distribution summary to return ['means', 'stds', 'q05', 'q95']
        name_prefix
            prefix to add to column names (f'{summary_name}{name_prefix}_{site_name}_{self\.factor_names_}')

        Returns
        -------
        Pandas data frame corresponding to either means, 5%/95% quantiles or sd of the posterior distribution

        """
        if type(self.factor_names_) is dict:
            factor_names_ = self.factor_names_[factor_names_key]
        else:
            factor_names_ = self.factor_names_

        return pd.DataFrame(
            samples[f"post_sample_{summary_name}"].get(site_name, None),
            index=self.adata.obs_names,
            columns=[f"{summary_name}{name_prefix}_{site_name}_{i}" for i in factor_names_],
        )

    def sample2df_vars(
        self,
        samples: dict,
        site_name: str = "gene_factors",
        summary_name: str = "means",
        name_prefix: str = "",
        factor_names_key: str = "",
    ):
        r"""Export posterior distribution summary for variable-specific parameters as Pandas data frame
        (means, 5%/95% quantiles or sd of posterior distribution).

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``
        site_name
            name of the model parameter to be exported
        summary_name
            posterior distribution summary to return ('means', 'stds', 'q05', 'q95')
        name_prefix
            prefix to add to column names (f'{summary_name}{name_prefix}_{site_name}_{self\.factor_names_}')

        Returns
        -------
        Pandas data frame corresponding to either means, 5%/95% quantiles or sd of the posterior distribution

        """
        if type(self.factor_names_) is dict:
            factor_names_ = self.factor_names_[factor_names_key]
        else:
            factor_names_ = self.factor_names_
        site = samples[f"post_sample_{summary_name}"].get(site_name, None)
        return pd.DataFrame(
            site,
            columns=self.adata.var_names,
            index=[f"{summary_name}{name_prefix}_{site_name}_{i}" for i in factor_names_],
        ).T

    def plot_QC(self, summary_name: str = "means", use_n_obs: int = 1000):
        """
        Show quality control plots:

        1. Reconstruction accuracy to assess if there are any issues with model training.
           The plot should be roughly diagonal, strong deviations signal problems that need to be investigated.
           Plotting is slow because expected value of mRNA count needs to be computed from model parameters. Random
           observations are used to speed up computation.

        Parameters
        ----------
        summary_name
            posterior distribution summary to use ('means', 'stds', 'q05', 'q95')

        Returns
        -------

        """

        if getattr(self, "samples", False) is False:
            raise RuntimeError("self.samples is missing, please run self.export_posterior() first")
        if use_n_obs is not None:
            ind_x = np.random.choice(
                self.adata_manager.adata.n_obs, np.min((use_n_obs, self.adata.n_obs)), replace=False
            )
        else:
            ind_x = None

        self.expected_nb_param = self.module.model.compute_expected(
            self.samples[f"post_sample_{summary_name}"], self.adata_manager, ind_x=ind_x
        )
        x_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)[ind_x, :]
        if issparse(x_data):
            x_data = np.asarray(x_data.toarray())
        self.plot_posterior_mu_vs_data(self.expected_nb_param["mu"], x_data)


class RegressionBackgroundDetectionTechPyroModel(PyroModule):
    r"""
    Given cell type annotation for each cell, the corresponding reference cell type signatures :math:`g_{f,g}`,
    which represent the average mRNA count of each gene `g` in each cell type `f={1, .., F}`,
    are estimated from sc/snRNA-seq data using Negative Binomial regression,
    which allows to robustly combine data across technologies and batches.

    This model combines batches, and treats data :math:`D` as Negative Binomial distributed,
    given mean :math:`\mu` and overdispersion :math:`\alpha`:

    .. math::
        D_{c,g} \sim \mathtt{NB}(alpha=\alpha_{g}, mu=\mu_{c,g})
    .. math::
        \mu_{c,g} = (\mu_{f,g} + s_{e,g}) * y_e * y_{t,g}

    Which is equivalent to:

    .. math::
        D_{c,g} \sim \mathtt{Poisson}(\mathtt{Gamma}(\alpha_{f,g}, \alpha_{f,g} / \mu_{c,g}))

    Here, :math:`\mu_{f,g}` denotes average mRNA count in each cell type :math:`f` for each gene :math:`g`;
    :math:`y_c` denotes normalisation for each experiment :math:`e` to account for  sequencing depth.
    :math:`y_{t,g}` denotes per gene :math:`g` detection efficiency normalisation for each technology :math:`t`.

    """

    def __init__(
        self,
        n_obs,
        n_vars,
        n_factors,
        n_batch,
        n_extra_categoricals=None,
        alpha_g_phi_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={
            "alpha": 1.0,
            "beta": 100.0,
        },
        detection_hyp_prior={"mean_alpha": 1.0, "mean_beta": 1.0},
        gene_tech_prior={"mean": 1, "alpha": 200},
        init_vals: Optional[dict] = None,
    ):
        """

        Parameters
        ----------
        n_obs
        n_vars
        n_factors
        n_batch
        n_extra_categoricals
        alpha_g_phi_hyp_prior
        gene_add_alpha_hyp_prior
        gene_add_mean_hyp_prior
        detection_hyp_prior
        gene_tech_prior
        """

        ############# Initialise parameters ################
        super().__init__()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_factors = n_factors
        self.n_batch = n_batch
        self.n_extra_categoricals = n_extra_categoricals

        self.alpha_g_phi_hyp_prior = alpha_g_phi_hyp_prior
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        self.detection_hyp_prior = detection_hyp_prior
        self.gene_tech_prior = gene_tech_prior

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))

        self.register_buffer(
            "detection_mean_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["mean_alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_beta",
            torch.tensor(self.detection_hyp_prior["mean_beta"]),
        )
        self.register_buffer(
            "gene_tech_prior_alpha",
            torch.tensor(self.gene_tech_prior["alpha"]),
        )
        self.register_buffer(
            "gene_tech_prior_beta",
            torch.tensor(self.gene_tech_prior["alpha"] / self.gene_tech_prior["mean"]),
        )

        self.register_buffer(
            "alpha_g_phi_hyp_prior_alpha",
            torch.tensor(self.alpha_g_phi_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "alpha_g_phi_hyp_prior_beta",
            torch.tensor(self.alpha_g_phi_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_alpha",
            torch.tensor(self.gene_add_alpha_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_beta",
            torch.tensor(self.gene_add_alpha_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_alpha",
            torch.tensor(self.gene_add_mean_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_beta",
            torch.tensor(self.gene_add_mean_hyp_prior["beta"]),
        )

        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("eps", torch.tensor(1e-8))

    ############# Define the model ################
    @staticmethod
    def _get_fn_args_from_batch_no_cat(tensor_dict):
        x_data = tensor_dict[REGISTRY_KEYS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY].squeeze(-1)
        # print('REGISTRY_KEYS.BATCH_KEY: ', REGISTRY_KEYS.BATCH_KEY)
        label_index = tensor_dict[REGISTRY_KEYS.LABELS_KEY].squeeze(-1)
        return (x_data, ind_x, batch_index, label_index, label_index), {}

    @staticmethod
    def _get_fn_args_from_batch_cat(tensor_dict):
        x_data = tensor_dict[REGISTRY_KEYS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY].squeeze(-1)
        label_index = tensor_dict[REGISTRY_KEYS.LABELS_KEY].squeeze(-1)
        extra_categoricals = tensor_dict[REGISTRY_KEYS.CAT_COVS_KEY]
        return (x_data, ind_x, batch_index, label_index, extra_categoricals), {}

    @property
    def _get_fn_args_from_batch(self):
        if self.n_extra_categoricals is not None:
            return self._get_fn_args_from_batch_cat
        else:
            return self._get_fn_args_from_batch_no_cat

    def create_plates(self, x_data, idx, batch_index, label_index, extra_categoricals):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

    def list_obs_plate_vars(self):
        """Create a dictionary with the name of observation/minibatch plate,
        indexes of model args to provide to encoder,
        variable names that belong to the observation plate
        and the number of dimensions in non-plate axis of each variable"""

        return {
            "name": "obs_plate",
            "input": [],  # expression data + (optional) batch index
            "input_transform": [],  # how to transform input data before passing to NN
            "sites": {},
        }

    def forward(self, x_data, idx, batch_index, label_index, extra_categoricals):
        # print('batch_index: ', batch_index)
        obs2sample = one_hot(batch_index, self.n_batch).float()
        # print("obs2sample.shape:", obs2sample.shape)
        # print('self.n_factors', self.n_factors)
        obs2label = one_hot(label_index, self.n_factors).float()
        # print('obs2label.shape: ', obs2label.shape)
        if self.n_extra_categoricals is not None:
            obs2extra_categoricals = torch.cat(
                [
                    one_hot(
                        extra_categoricals[:, i].view((extra_categoricals.shape[0], 1)),
                        n_cat,
                    )
                    for i, n_cat in enumerate(self.n_extra_categoricals)
                ],
                dim=1,
            )

        obs_plate = self.create_plates(x_data, idx, batch_index, label_index, extra_categoricals)

        # =====================Per-cluster average mRNA count ======================= #
        # \mu_{f,g}
        per_cluster_mu_fg = pyro.sample(
            "per_cluster_mu_fg",
            dist.Gamma(self.ones, self.ones).expand([self.n_factors, self.n_vars]).to_event(2),
        )

        # =====================Gene-specific multiplicative component ======================= #
        # `y_{t, g}` per gene multiplicative effect that explains the difference
        # in sensitivity between genes in each technology or covariate effect
        if self.n_extra_categoricals is not None:
            detection_tech_gene_tg = pyro.sample(
                "detection_tech_gene_tg",
                dist.Gamma(
                    self.ones * self.gene_tech_prior_alpha,
                    self.ones * self.gene_tech_prior_beta,
                )
                .expand([np.sum(self.n_extra_categoricals), self.n_vars])
                .to_event(2),
            )

        # =====================Cell-specific detection efficiency ======================= #
        # y_c with hierarchical mean prior
        detection_mean_y_e = pyro.sample(
            "detection_mean_y_e",
            dist.Gamma(
                self.ones * self.detection_mean_hyp_prior_alpha,
                self.ones * self.detection_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )
        # obs2sample = obs2sample.float()
        detection_y_c = obs2sample @ detection_mean_y_e  # (self.n_obs, 1)
        detection_y_c = detection_y_c.view(-1, 1)
        # print('detection_y_c.shape: ', detection_y_c.shape)

        # =====================Gene-specific additive component ======================= #
        # s_{e,g} accounting for background, free-floating RNA
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            dist.Gamma(self.ones * self.gene_add_alpha_hyp_prior_alpha, self.ones * self.gene_add_alpha_hyp_prior_beta),
        )
        s_g_gene_add_mean = pyro.sample(
            "s_g_gene_add_mean",
            dist.Gamma(
                self.gene_add_mean_hyp_prior_alpha,
                self.gene_add_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )  # (self.n_batch)
        s_g_gene_add_alpha_e_inv = pyro.sample(
            "s_g_gene_add_alpha_e_inv",
            dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1]).to_event(2),
        )  # (self.n_batch)
        s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)

        s_g_gene_add = pyro.sample(
            "s_g_gene_add",
            dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean)
            .expand([self.n_batch, self.n_vars])
            .to_event(2),
        )  # (self.n_batch, n_vars)
        # print("s_g_gene_add.shape:", s_g_gene_add.shape)

        # =====================Gene-specific overdispersion ======================= #
        alpha_g_phi_hyp = pyro.sample(
            "alpha_g_phi_hyp",
            dist.Gamma(self.ones * self.alpha_g_phi_hyp_prior_alpha, self.ones * self.alpha_g_phi_hyp_prior_beta),
        )
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([1, self.n_vars]).to_event(2),
        )  # (self.n_batch or 1, self.n_vars)

        # =====================Expected expression ======================= #

        # overdispersion
        alpha = self.ones / alpha_g_inverse.pow(2)
        # biological expression
        mu = (
            obs2label @ per_cluster_mu_fg + obs2sample @ s_g_gene_add  # contaminating RNA
        ) * detection_y_c  # cell-specific normalisation
        # print("mu.shape after detection_y_c:", mu.shape)
        if self.n_extra_categoricals is not None:
            # gene-specific normalisation for covatiates
            mu = mu * (obs2extra_categoricals @ detection_tech_gene_tg)
        # total_count, logits = _convert_mean_disp_to_counts_logits(
        #    mu, alpha, eps=self.eps
        # )

        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        with obs_plate:
            # print("rate.shape:", (alpha / mu).shape)
            # print("data.shape:", x_data.shape)
            pyro.sample(
                "data_target",
                dist.GammaPoisson(concentration=alpha, rate=alpha / mu),
                # dist.NegativeBinomial(total_count=total_count, logits=logits),
                obs=x_data,
            )

    # =====================Other functions======================= #
    def compute_expected(self, samples, adata_manager, ind_x=None):
        r"""Compute expected expression of each gene in each cell. Useful for evaluating how well
        the model learned expression pattern of all genes in the data.

        Parameters
        ----------
        samples
            dictionary with values of the posterior
        adata
            registered anndata
        ind_x
            indices of cells to use (to reduce data size)
        """
        if ind_x is None:
            ind_x = np.arange(adata_manager.adata.n_obs).astype(int)
        else:
            ind_x = ind_x.astype(int)
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten()).values[ind_x, :].astype("float32")
        obs2label = adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
        obs2label = pd.get_dummies(obs2label.flatten()).values[ind_x, :].astype("float32")
        if self.n_extra_categoricals is not None:
            extra_categoricals = adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            obs2extra_categoricals = np.concatenate(
                [
                    pd.get_dummies(extra_categoricals.iloc[ind_x, i]).astype("float32")
                    for i, n_cat in enumerate(self.n_extra_categoricals)
                ],
                axis=1,
            )

        alpha = 1 / np.power(samples["alpha_g_inverse"], 2)

        mu = (np.dot(obs2label, samples["per_cluster_mu_fg"]) + np.dot(obs2sample, samples["s_g_gene_add"])) * np.dot(
            obs2sample, samples["detection_mean_y_e"]
        )  # samples["detection_y_c"][ind_x, :]
        if self.n_extra_categoricals is not None:
            mu = mu * np.dot(obs2extra_categoricals, samples["detection_tech_gene_tg"])

        return {"mu": mu, "alpha": alpha}

    def compute_expected_subset(self, samples, adata_manager, fact_ind, cell_ind):
        r"""Compute expected expression of each gene in each cell that comes from
        a subset of factors (cell types) or cells.

        Useful for evaluating how well the model learned expression pattern of all genes in the data.

        Parameters
        ----------
        samples
            dictionary with values of the posterior
        adata
            registered anndata
        fact_ind
            indices of factors/cell types to use
        cell_ind
            indices of cells to use
        """
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten())
        obs2label = adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
        obs2label = pd.get_dummies(obs2label.flatten())
        if self.n_extra_categoricals is not None:
            extra_categoricals = adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            obs2extra_categoricals = np.concatenate(
                [pd.get_dummies(extra_categoricals.iloc[:, i]) for i, n_cat in enumerate(self.n_extra_categoricals)],
                axis=1,
            )

        alpha = 1 / np.power(samples["alpha_g_inverse"], 2)

        mu = (
            np.dot(obs2label[cell_ind, fact_ind], samples["per_cluster_mu_fg"][fact_ind, :])
            + np.dot(obs2sample[cell_ind, :], samples["s_g_gene_add"])
        ) * np.dot(
            obs2sample, samples["detection_mean_y_e"]
        )  # samples["detection_y_c"]
        if self.n_extra_categoricals is not None:
            mu = mu * np.dot(obs2extra_categoricals[cell_ind, :], samples["detection_tech_gene_tg"])

        return {"mu": mu, "alpha": alpha}

    def normalise(self, samples, adata_manager, adata):
        r"""Normalise expression data by estimated technical variables.

        Parameters
        ----------
        samples
            dictionary with values of the posterior
        adata
            registered anndata

        """
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten())
        if self.n_extra_categoricals is not None:
            extra_categoricals = adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            obs2extra_categoricals = np.concatenate(
                [pd.get_dummies(extra_categoricals.iloc[:, i]) for i, n_cat in enumerate(self.n_extra_categoricals)],
                axis=1,
            )
        # get counts matrix
        corrected = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        # normalise per-sample scaling
        corrected = corrected / np.dot(obs2sample, samples["detection_mean_y_e"])
        # normalise per gene effects
        if self.n_extra_categoricals is not None:
            corrected = corrected / np.dot(obs2extra_categoricals, samples["detection_tech_gene_tg"])

        # remove additive sample effects
        corrected = corrected - np.dot(obs2sample, samples["s_g_gene_add"])

        # set minimum value to 0 for each gene (a hack to avoid negative values)
        corrected = corrected - corrected.min()

        return corrected


def init_to_value(site=None, values={}, init_fn=init_to_mean):
    if site is None:
        return partial(init_to_value, values=values)
    if site["name"] in values:
        return values[site["name"]]
    else:
        return init_fn(site)


class AutoGuideMixinModule:
    """
    This mixin class provides methods for:

    - initialising standard AutoNormal guides
    - initialising amortised guides (AutoNormalEncoder)
    - initialising amortised guides with special additional inputs

    """

    def _create_autoguide(
        self,
        model,
        amortised,
        encoder_kwargs,
        data_transform,
        encoder_mode,
        init_loc_fn=init_to_mean(fallback=init_to_feasible),
        n_cat_list: list = [],
        encoder_instance=None,
        guide_class=AutoNormal,
        guide_kwargs: Optional[dict] = None,
    ):

        if guide_kwargs is None:
            guide_kwargs = dict()

        if not amortised:
            if getattr(model, "discrete_variables", None) is not None:
                model = poutine.block(model, hide=model.discrete_variables)
            if issubclass(guide_class, poutine.messenger.Messenger):
                # messenger guides don't need create_plates function
                _guide = guide_class(
                    model,
                    init_loc_fn=init_loc_fn,
                    **guide_kwargs,
                )
            else:
                _guide = guide_class(
                    model,
                    init_loc_fn=init_loc_fn,
                    **guide_kwargs,
                    create_plates=self.model.create_plates,
                )
        else:
            encoder_kwargs = encoder_kwargs if isinstance(encoder_kwargs, dict) else dict()
            n_hidden = encoder_kwargs["n_hidden"] if "n_hidden" in encoder_kwargs.keys() else 200
            if data_transform is None:
                pass
            elif isinstance(data_transform, np.ndarray):
                # add extra info about gene clusters as input to NN
                self.register_buffer("gene_clusters", torch.tensor(data_transform.astype("float32")))
                n_in = model.n_vars + data_transform.shape[1]
                data_transform = self._data_transform_clusters()
            elif data_transform == "log1p":
                # use simple log1p transform
                data_transform = torch.log1p
                n_in = self.model.n_vars
            elif (
                isinstance(data_transform, dict)
                and "var_std" in list(data_transform.keys())
                and "var_mean" in list(data_transform.keys())
            ):
                # use data transform by scaling
                n_in = model.n_vars
                self.register_buffer(
                    "var_mean",
                    torch.tensor(data_transform["var_mean"].astype("float32").reshape((1, n_in))),
                )
                self.register_buffer(
                    "var_std",
                    torch.tensor(data_transform["var_std"].astype("float32").reshape((1, n_in))),
                )
                data_transform = self._data_transform_scale()
            else:
                # use custom data transform
                data_transform = data_transform
                n_in = model.n_vars
            amortised_vars = model.list_obs_plate_vars()
            if len(amortised_vars["input"]) >= 2:
                encoder_kwargs["n_cat_list"] = n_cat_list
            if data_transform is not None:
                amortised_vars["input_transform"][0] = data_transform
            if "n_in" in amortised_vars.keys():
                n_in = amortised_vars["n_in"]
            if getattr(model, "discrete_variables", None) is not None:
                model = poutine.block(model, hide=model.discrete_variables)
            _guide = AutoAmortisedHierarchicalNormalMessenger(
                model,
                amortised_plate_sites=amortised_vars,
                n_in=n_in,
                n_hidden=n_hidden,
                encoder_kwargs=encoder_kwargs,
                encoder_mode=encoder_mode,
                encoder_instance=encoder_instance,
                init_loc_fn=init_loc_fn,
                **guide_kwargs,
            )
        return _guide

    def _data_transform_clusters(self):
        def _data_transform(x):
            return torch.log1p(torch.cat([x, x @ self.gene_clusters], dim=1))

        return _data_transform

    def _data_transform_scale(self):
        def _data_transform(x):
            # return (x - self.var_mean) / self.var_std
            return x / self.var_std

        return _data_transform


class RegressionBaseModule(PyroBaseModuleClass, AutoGuideMixinModule):
    def __init__(
        self,
        model,
        amortised: bool = False,
        encoder_mode: Literal["single", "multiple", "single-multiple"] = "single",
        encoder_kwargs=None,
        data_transform="log1p",
        **kwargs,
    ):
        """
        Module class which defines AutoGuide given model. Supports multiple model architectures.

        Parameters
        ----------
        amortised
            boolean, use a Neural Network to approximate posterior distribution of location-specific (local) parameters?
        encoder_mode
            Use single encoder for all variables ("single"), one encoder per variable ("multiple")
            or a single encoder in the first step and multiple encoders in the second step ("single-multiple").
        encoder_kwargs
            arguments for Neural Network construction (scvi.nn.FCLayers)
        kwargs
            arguments for specific model class - e.g. number of genes, values of the prior distribution
        """
        super().__init__()
        self.hist = []

        self._model = model(**kwargs)
        self._amortised = amortised

        self._guide = self._create_autoguide(
            model=self.model,
            amortised=self.is_amortised,
            encoder_kwargs=encoder_kwargs,
            data_transform=data_transform,
            encoder_mode=encoder_mode,
            init_loc_fn=self.init_to_value,
            n_cat_list=[kwargs["n_batch"]],
        )

        self._get_fn_args_from_batch = self._model._get_fn_args_from_batch

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    @property
    def is_amortised(self):
        return self._amortised

    @property
    def list_obs_plate_vars(self):
        return self.model.list_obs_plate_vars()

    def init_to_value(self, site):

        if getattr(self.model, "np_init_vals", None) is not None:
            init_vals = {k: getattr(self.model, f"init_val_{k}") for k in self.model.np_init_vals.keys()}
        else:
            init_vals = dict()
        return init_to_value(site=site, values=init_vals)


def compute_cluster_averages(adata, labels, use_raw=True, layer=None):
    """
    Compute average expression of each gene in each cluster

    Parameters
    ----------
    adata
        AnnData object of reference single-cell dataset
    labels
        Name of adata.obs column containing cluster labels
    use_raw
        Use raw slow in adata.
    layer
        Use layer in adata, provide layer name.

    Returns
    -------
    pd.DataFrame of cluster average expression of each gene

    """

    if layer is not None:
        x = adata.layers[layer]
        var_names = adata.var_names
    else:
        if not use_raw:
            x = adata.X
            var_names = adata.var_names
        else:
            if not adata.raw:
                raise ValueError("AnnData object has no raw data, change `use_raw=True, layer=None` or fix your object")
            x = adata.raw.X
            var_names = adata.raw.var_names

    if sum(adata.obs.columns == labels) != 1:
        raise ValueError("`labels` is absent in adata_ref.obs or not unique")

    all_clusters = np.unique(adata.obs[labels])
    averages_mat = np.zeros((1, x.shape[1]))

    for c in all_clusters:
        sparse_subset = csr_matrix(x[np.isin(adata.obs[labels], c), :])
        aver = sparse_subset.mean(0)
        averages_mat = np.concatenate((averages_mat, aver))
    averages_mat = averages_mat[1:, :].T
    averages_df = pd.DataFrame(data=averages_mat, index=var_names, columns=all_clusters)

    return averages_df


class RegressionModel(QuantileMixin, PyroSampleMixin, PyroSviTrainMixin, PltExportMixin, BaseModelClass):
    """
    Model which estimates per cluster average mRNA count account for batch effects. User-end model class.

    https://github.com/BayraktarLab/cell2location

    Parameters
    ----------
    adata
        single-cell AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    use_gpu
        Use the GPU?
    **model_kwargs
        Keyword args for :class:`~scvi.external.LocationModelLinearDependentWMultiExperimentModel`

    Examples
    --------
    TODO add example
    >>>
    """

    def __init__(
        self,
        adata: AnnData,
        model_class=None,
        use_average_as_initial: bool = True,
        **model_kwargs,
    ):
        # in case any other model was created before that shares the same parameter names.
        clear_param_store()

        super().__init__(adata)

        if model_class is None:
            model_class = RegressionBackgroundDetectionTechPyroModel

        # annotations for cell types
        self.n_factors_ = self.summary_stats["n_labels"]
        # print('self.n_factors_', self.n_factors_)
        self.factor_names_ = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping
        # annotations for extra categorical covariates
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            self.extra_categoricals_ = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            self.n_extra_categoricals_ = self.extra_categoricals_.n_cats_per_key
            model_kwargs["n_extra_categoricals"] = self.n_extra_categoricals_

        # use per class average as initial value
        if use_average_as_initial:
            # compute cluster average expression
            aver = self._compute_cluster_averages(key=REGISTRY_KEYS.LABELS_KEY)
            model_kwargs["init_vals"] = {"per_cluster_mu_fg": aver.values.T.astype("float32") + 0.0001}

        # print('model_kwargs: ', model_kwargs)
        self.module = RegressionBaseModule(
            model=model_class,
            n_obs=self.summary_stats["n_cells"],
            n_vars=self.summary_stats["n_vars"],
            n_factors=self.n_factors_,
            n_batch=self.summary_stats["n_batch"],
            **model_kwargs,
        )
        self._model_summary_string = f'RegressionBackgroundDetectionTech model with the following params: \nn_factors: {self.n_factors_} \nn_batch: {self.summary_stats["n_batch"]} '
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        # print('batch_key: ', batch_key)
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs).astype("int64")
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        # print('anndata_fields: ', anndata_fields)
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        # print('adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY).shape: ', adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY).shape)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: Optional[int] = None,
        batch_size: int = 2500,
        train_size: float = 1,
        lr: float = 0.002,
        **kwargs,
    ):
        """Train the model with useful defaults

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        train_size
            Size of training set in the range [0.0, 1.0].
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        kwargs
            Other arguments to scvi.model.base.PyroSviTrainMixin().train() method
        """

        kwargs["max_epochs"] = max_epochs
        kwargs["batch_size"] = batch_size
        kwargs["train_size"] = train_size
        kwargs["lr"] = lr

        super().train(**kwargs)

    def _compute_cluster_averages(self, key=REGISTRY_KEYS.LABELS_KEY):
        """
        Compute average per cluster (key=REGISTRY_KEYS.LABELS_KEY) or per batch (key=REGISTRY_KEYS.BATCH_KEY).

        Returns
        -------
        pd.DataFrame with variables in rows and labels in columns
        """
        # find cell label column
        label_col = self.adata_manager.get_state_registry(key).original_key

        # find data slot
        x_dict = self.adata_manager.data_registry["X"]
        if x_dict["attr_name"] == "X":
            use_raw = False
        else:
            use_raw = True
        if x_dict["attr_name"] == "layers":
            layer = x_dict["attr_key"]
        else:
            layer = None

        # compute mean expression of each gene in each cluster/batch
        aver = compute_cluster_averages(self.adata, labels=label_col, use_raw=use_raw, layer=layer)

        return aver

    def export_posterior(
        self,
        adata,
        sample_kwargs: Optional[dict] = None,
        export_slot: str = "mod",
        add_to_varm: list = ["means", "stds", "q05", "q95"],
        scale_average_detection: bool = True,
        use_quantiles: bool = False,
    ):
        """
        Summarise posterior distribution and export results (cell abundance) to anndata object:
        1. adata.obsm: Estimated references expression signatures (average mRNA count in each cell type),
            as pd.DataFrames for each posterior distribution summary `add_to_varm`,
            posterior mean, sd, 5% and 95% quantiles (['means', 'stds', 'q05', 'q95']).
            If export to adata.varm fails with error, results are saved to adata.var instead.
        2. adata.uns: Posterior of all parameters, model name, date,
            cell type names ('factor_names'), obs and var names.

        Parameters
        ----------
        adata
            anndata object where results should be saved
        sample_kwargs
            arguments for self.sample_posterior (generating and summarising posterior samples), namely:
                num_samples - number of samples to use (Default = 1000).
                batch_size - data batch size (keep low enough to fit on GPU, default 2048).
                use_gpu - use gpu for generating samples?
        export_slot
            adata.uns slot where to export results
        add_to_varm
            posterior distribution summary to export in adata.varm (['means', 'stds', 'q05', 'q95']).
        use_quantiles
            compute quantiles directly (True, more memory efficient) or use samples (False, default).
            If True, means and stds cannot be computed so are not exported and returned.
        Returns
        -------

        """

        sample_kwargs = sample_kwargs if isinstance(sample_kwargs, dict) else dict()

        # get posterior distribution summary
        if use_quantiles:
            add_to_varm = [i for i in add_to_varm if (i not in ["means", "stds"]) and ("q" in i)]
            if len(add_to_varm) == 0:
                raise ValueError("No quantiles to export - please add add_to_obsm=['q05', 'q50', 'q95'].")
            self.samples = dict()
            for i in add_to_varm:
                q = float(f"0.{i[1:]}")
                self.samples[f"post_sample_{i}"] = self.posterior_quantile(q=q, **sample_kwargs)
        else:
            # generate samples from posterior distributions for all parameters
            # and compute mean, 5%/95% quantiles and standard deviation
            self.samples = self.sample_posterior(**sample_kwargs)

        # export posterior distribution summary for all parameters and
        # annotation (model, date, var, obs and cell type names) to anndata object
        adata.uns[export_slot] = self._export2adata(self.samples)

        # export estimated expression in each cluster
        # first convert np.arrays to pd.DataFrames with cell type and observation names
        # data frames contain mean, 5%/95% quantiles and standard deviation, denoted by a prefix
        for k in add_to_varm:
            sample_df = self.sample2df_vars(
                self.samples,
                site_name="per_cluster_mu_fg",
                summary_name=k,
                name_prefix="",
            )
            if scale_average_detection and ("detection_y_c" in list(self.samples[f"post_sample_{k}"].keys())):
                sample_df = sample_df * self.samples[f"post_sample_{k}"]["detection_y_c"].mean()
            try:
                adata.varm[f"{k}_per_cluster_mu_fg"] = sample_df.loc[adata.var.index, :]
            except ValueError:
                # Catching weird error with obsm: `ValueError: value.index does not match parent’s axis 1 names`
                adata.var[sample_df.columns] = sample_df.loc[adata.var.index, :]

        return adata

    def plot_QC(
        self,
        summary_name: str = "means",
        use_n_obs: int = 1000,
        scale_average_detection: bool = True,
    ):
        """
        Show quality control plots:
        1. Reconstruction accuracy to assess if there are any issues with model training.
            The plot should be roughly diagonal, strong deviations signal problems that need to be investigated.
            Plotting is slow because expected value of mRNA count needs to be computed from model parameters. Random
            observations are used to speed up computation.

        2. Estimated reference expression signatures (accounting for batch effect)
            compared to average expression in each cluster. We expect the signatures to be different
            from average when batch effects are present, however, when this plot is very different from
            a perfect diagonal, such as very low values on Y-axis, non-zero density everywhere)
            it indicates problems with signature estimation.

        Parameters
        ----------
        summary_name
            posterior distribution summary to use ('means', 'stds', 'q05', 'q95')

        Returns
        -------

        """

        super().plot_QC(summary_name=summary_name, use_n_obs=use_n_obs)
        plt.show()

        inf_aver = self.samples[f"post_sample_{summary_name}"]["per_cluster_mu_fg"].T
        if scale_average_detection and ("detection_y_c" in list(self.samples[f"post_sample_{summary_name}"].keys())):
            inf_aver = inf_aver * self.samples[f"post_sample_{summary_name}"]["detection_y_c"].mean()
        aver = self._compute_cluster_averages(key=REGISTRY_KEYS.LABELS_KEY)
        aver = aver[self.factor_names_]

        plt.hist2d(
            np.log10(aver.values.flatten() + 1),
            np.log10(inf_aver.flatten() + 1),
            bins=50,
            norm=matplotlib.colors.LogNorm(),
        )
        plt.xlabel("Mean expression for every gene in every cluster")
        plt.ylabel("Estimated expression for every gene in every cluster")
        plt.show()


###################################################################
#########################Cell2location#############################
###################################################################


class LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel(PyroModule):
    r"""
    Cell2location models the elements of :math:`D` as Negative Binomial distributed,
    given an unobserved gene expression level (rate) :math:`mu` and a gene- and batch-specific
    over-dispersion parameter :math:`\alpha_{e,g}` which accounts for unexplained variance:

    .. math::
        D_{s,g} \sim \mathtt{NB}(\mu_{s,g}, \alpha_{e,g})

    The expression level of genes :math:`\mu_{s,g}` in the mRNA count space is modelled
    as a linear function of expression signatures of reference cell types :math:`g_{f,g}`:

    .. math::
        \mu_{s,g} = (m_{g} \left (\sum_{f} {w_{s,f} \: g_{f,g}} \right) + s_{e,g}) y_{s}

    Here, :math:`w_{s,f}` denotes regression weight of each reference signature :math:`f` at location :math:`s`, which can be interpreted as the expected number of cells at location :math:`s` that express reference signature :math:`f`;
    :math:`g_{f,g}` denotes the reference signatures of cell types :math:`f` of each gene :math:`g`, `cell_state_df` input ;
    :math:`m_{g}` denotes a gene-specific scaling parameter which adjusts for global differences in sensitivity between technologies (platform effect);
    :math:`y_{s}` denotes a location/observation-specific scaling parameter which adjusts for differences in sensitivity between observations and batches;
    :math:`s_{e,g}` is additive component that account for gene- and location-specific shift, such as due to contaminating or free-floating RNA.

    To account for the similarity of location patterns across cell types, :math:`w_{s,f}` is modelled using
    another layer  of decomposition (factorization) using :math:`r={1, .., R}` groups of cell types,
    that can be interpreted as cellular compartments or tissue zones. Unless stated otherwise, R is set to 50.

    Corresponding graphical model can be found in supplementary methods:
    https://www.biorxiv.org/content/10.1101/2020.11.15.378125v1.supplementary-material

    Approximate Variational Inference is used to estimate the posterior distribution of all model parameters.

    Estimation of absolute cell abundance :math:`w_{s,f}` is guided using informed prior on the number of cells
    (argument called `N_cells_per_location`). It is a tissue-level global estimate, which can be derived from histology
    images (H&E or DAPI), ideally paired to the spatial expression data or at least representing the same tissue type.
    This parameter can be estimated by manually counting nuclei in a 10-20 locations in the histology image
    (e.g. using 10X Loupe browser), and computing the average cell abundance.
    An appropriate setting of this prior is essential to inform the estimation of absolute cell type abundance values,
    however, the model is robust to a range of similar values.
    In settings where suitable histology images are not available, the size of capture regions relative to
    the expected size of cells can be used to estimate `N_cells_per_location`.

    The prior on detection efficiency per location :math:`y_s` is selected to discourage over-normalisation, such that
    unless data has evidence of strong technical effect, the effect is assumed to be small and close to
    the mean sensitivity for each batch :math:`y_e`:

    .. math::
        y_s \sim Gamma(detection\_alpha, detection\_alpha / y_e)

    where y_e is unknown/latent average detection efficiency in each batch/experiment:

    .. math::
        y_e \sim Gamma(10, 10 / detection\_mean)

    """

    # training mode without observed data (just using priors)
    training_wo_observed = False
    training_wo_initial = False

    def __init__(
        self,
        n_obs,
        n_vars,
        n_factors,
        n_batch,
        cell_state_mat,
        n_groups: int = 50,
        detection_mean=1 / 2,
        detection_alpha=20.0,
        m_g_gene_level_prior={"mean": 1, "mean_var_ratio": 1.0, "alpha_mean": 3.0},
        N_cells_per_location=8.0,
        A_factors_per_location=7.0,
        B_groups_per_location=7.0,
        N_cells_mean_var_ratio=1.0,
        alpha_g_phi_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={
            "alpha": 1.0,
            "beta": 100.0,
        },
        detection_hyp_prior={"mean_alpha": 10.0},
        w_sf_mean_var_ratio=5.0,
        init_vals: Optional[dict] = None,
        init_alpha=20.0,
        dropout_p=0.0,
    ):
        super().__init__()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_factors = n_factors
        self.n_batch = n_batch
        self.n_groups = n_groups

        self.m_g_gene_level_prior = m_g_gene_level_prior

        self.alpha_g_phi_hyp_prior = alpha_g_phi_hyp_prior
        self.w_sf_mean_var_ratio = w_sf_mean_var_ratio
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        detection_hyp_prior["mean"] = detection_mean
        detection_hyp_prior["alpha"] = detection_alpha
        self.detection_hyp_prior = detection_hyp_prior

        self.dropout_p = dropout_p
        if self.dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=self.dropout_p)

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))
            self.init_alpha = init_alpha
            self.register_buffer("init_alpha_tt", torch.tensor(self.init_alpha))

        factors_per_groups = A_factors_per_location / B_groups_per_location

        self.register_buffer(
            "detection_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["mean_alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_beta",
            torch.tensor(self.detection_hyp_prior["mean_alpha"] / self.detection_hyp_prior["mean"]),
        )

        # compute hyperparameters from mean and sd
        self.register_buffer("m_g_mu_hyp", torch.tensor(self.m_g_gene_level_prior["mean"]))
        self.register_buffer(
            "m_g_mu_mean_var_ratio_hyp",
            torch.tensor(self.m_g_gene_level_prior["mean_var_ratio"]),
        )

        self.register_buffer("m_g_alpha_hyp_mean", torch.tensor(self.m_g_gene_level_prior["alpha_mean"]))

        self.cell_state_mat = cell_state_mat
        self.register_buffer("cell_state", torch.tensor(cell_state_mat.T))

        self.register_buffer("N_cells_per_location", torch.tensor(N_cells_per_location))
        self.register_buffer("factors_per_groups", torch.tensor(factors_per_groups))
        self.register_buffer("B_groups_per_location", torch.tensor(B_groups_per_location))
        self.register_buffer("N_cells_mean_var_ratio", torch.tensor(N_cells_mean_var_ratio))

        self.register_buffer(
            "alpha_g_phi_hyp_prior_alpha",
            torch.tensor(self.alpha_g_phi_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "alpha_g_phi_hyp_prior_beta",
            torch.tensor(self.alpha_g_phi_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_alpha",
            torch.tensor(self.gene_add_alpha_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_beta",
            torch.tensor(self.gene_add_alpha_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_alpha",
            torch.tensor(self.gene_add_mean_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_beta",
            torch.tensor(self.gene_add_mean_hyp_prior["beta"]),
        )

        self.register_buffer("w_sf_mean_var_ratio_tensor", torch.tensor(self.w_sf_mean_var_ratio))

        self.register_buffer("n_factors_tensor", torch.tensor(self.n_factors))
        self.register_buffer("n_groups_tensor", torch.tensor(self.n_groups))

        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("ones_1_n_groups", torch.ones((1, self.n_groups)))
        self.register_buffer("ones_n_batch_1", torch.ones((self.n_batch, 1)))
        self.register_buffer("eps", torch.tensor(1e-8))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x_data = tensor_dict[REGISTRY_KEYS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY].squeeze(-1)
        return (x_data, ind_x, batch_index), {}

    def create_plates(self, x_data, idx, batch_index):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

    def list_obs_plate_vars(self):
        """
        Create a dictionary with:

        1. "name" - the name of observation/minibatch plate;
        2. "input" - indexes of model args to provide to encoder network when using amortised inference;
        3. "sites" - dictionary with

          * keys - names of variables that belong to the observation plate (used to recognise
            and merge posterior samples for minibatch variables)
          * values - the dimensions in non-plate axis of each variable (used to construct output
            layer of encoder network when using amortised inference)
        """

        return {
            "name": "obs_plate",
            "input": [0, 2],  # expression data + (optional) batch index
            "input_transform": [
                torch.log1p,
                lambda x: x,
            ],  # how to transform input data before passing to NN
            "input_normalisation": [
                False,
                False,
            ],  # whether to normalise input data before passing to NN
            "sites": {
                "n_s_cells_per_location": 1,
                "b_s_groups_per_location": 1,
                "z_sr_groups_factors": self.n_groups,
                "w_sf": self.n_factors,
                "detection_y_s": 1,
            },
        }

    def forward(self, x_data, idx, batch_index):
        obs2sample = one_hot(batch_index, self.n_batch).float()

        obs_plate = self.create_plates(x_data, idx, batch_index)

        # =====================Gene expression level scaling m_g======================= #
        # Explains difference in sensitivity for each gene between single cell and spatial technology
        m_g_mean = pyro.sample(
            "m_g_mean",
            dist.Gamma(
                self.m_g_mu_mean_var_ratio_hyp * self.m_g_mu_hyp,
                self.m_g_mu_mean_var_ratio_hyp,
            )
            .expand([1, 1])
            .to_event(2),
        )  # (1, 1)

        m_g_alpha_e_inv = pyro.sample(
            "m_g_alpha_e_inv",
            dist.Exponential(self.m_g_alpha_hyp_mean).expand([1, 1]).to_event(2),
        )  # (1, 1)
        m_g_alpha_e = self.ones / m_g_alpha_e_inv.pow(2)

        m_g = pyro.sample(
            "m_g",
            dist.Gamma(m_g_alpha_e, m_g_alpha_e / m_g_mean).expand([1, self.n_vars]).to_event(2),  # self.m_g_mu_hyp)
        )  # (1, n_vars)

        # =====================Cell abundances w_sf======================= #
        # factorisation prior on w_sf models similarity in locations
        # across cell types f and reflects the absolute scale of w_sf
        with obs_plate as ind:
            k = "n_s_cells_per_location"
            n_s_cells_per_location = pyro.sample(
                k,
                dist.Gamma(
                    self.N_cells_per_location * self.N_cells_mean_var_ratio,
                    self.N_cells_mean_var_ratio,
                ),
            )
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                # pre-training Variational distribution to initial values
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=n_s_cells_per_location,
                )  # (self.n_obs, self.n_groups)

            k = "b_s_groups_per_location"
            b_s_groups_per_location = pyro.sample(
                k,
                dist.Gamma(self.B_groups_per_location, self.ones),
            )
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                # pre-training Variational distribution to initial values
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=b_s_groups_per_location,
                )  # (self.n_obs, self.n_groups)

        # cell group loadings
        shape = self.ones_1_n_groups * b_s_groups_per_location / self.n_groups_tensor
        rate = self.ones_1_n_groups / (n_s_cells_per_location / b_s_groups_per_location)
        with obs_plate as ind:
            k = "z_sr_groups_factors"
            z_sr_groups_factors = pyro.sample(
                k,
                dist.Gamma(shape, rate),  # .to_event(1)#.expand([self.n_groups]).to_event(1)
            )  # (n_obs, n_groups)

            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                # pre-training Variational distribution to initial values
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=z_sr_groups_factors,
                )  # (self.n_obs, self.n_groups)

        k_r_factors_per_groups = pyro.sample(
            "k_r_factors_per_groups",
            dist.Gamma(self.factors_per_groups, self.ones).expand([self.n_groups, 1]).to_event(2),
        )  # (self.n_groups, 1)

        c2f_shape = k_r_factors_per_groups / self.n_factors_tensor

        x_fr_group2fact = pyro.sample(
            "x_fr_group2fact",
            dist.Gamma(c2f_shape, k_r_factors_per_groups).expand([self.n_groups, self.n_factors]).to_event(2),
        )  # (self.n_groups, self.n_factors)

        with obs_plate as ind:
            w_sf_mu = z_sr_groups_factors @ x_fr_group2fact

            k = "w_sf"
            w_sf = pyro.sample(
                k,
                dist.Gamma(
                    w_sf_mu * self.w_sf_mean_var_ratio_tensor,
                    self.w_sf_mean_var_ratio_tensor,
                ),
            )  # (self.n_obs, self.n_factors)
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                # pre-training Variational distribution to initial values
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=w_sf,
                )  # (self.n_obs, self.n_factors)

        # =====================Location-specific detection efficiency ======================= #
        # y_s with hierarchical mean prior
        detection_mean_y_e = pyro.sample(
            "detection_mean_y_e",
            dist.Gamma(
                self.ones * self.detection_mean_hyp_prior_alpha,
                self.ones * self.detection_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )
        detection_hyp_prior_alpha = pyro.deterministic(
            "detection_hyp_prior_alpha",
            self.ones_n_batch_1 * self.detection_hyp_prior_alpha,
        )

        beta = (obs2sample @ detection_hyp_prior_alpha) / (obs2sample @ detection_mean_y_e)
        with obs_plate:
            k = "detection_y_s"
            detection_y_s = pyro.sample(
                k,
                dist.Gamma(obs2sample @ detection_hyp_prior_alpha, beta),
            )  # (self.n_obs, 1)

            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                # pre-training Variational distribution to initial values
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=detection_y_s,
                )  # (self.n_obs, 1)

        # =====================Gene-specific additive component ======================= #
        # per gene molecule contribution that cannot be explained by
        # cell state signatures (e.g. background, free-floating RNA)
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            dist.Gamma(self.ones * self.gene_add_alpha_hyp_prior_alpha, self.ones * self.gene_add_alpha_hyp_prior_beta),
        )
        s_g_gene_add_mean = pyro.sample(
            "s_g_gene_add_mean",
            dist.Gamma(
                self.gene_add_mean_hyp_prior_alpha,
                self.gene_add_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )  # (self.n_batch)
        s_g_gene_add_alpha_e_inv = pyro.sample(
            "s_g_gene_add_alpha_e_inv",
            dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1]).to_event(2),
        )  # (self.n_batch)
        s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)

        s_g_gene_add = pyro.sample(
            "s_g_gene_add",
            dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean)
            .expand([self.n_batch, self.n_vars])
            .to_event(2),
        )  # (self.n_batch, n_vars)

        # =====================Gene-specific overdispersion ======================= #
        alpha_g_phi_hyp = pyro.sample(
            "alpha_g_phi_hyp",
            dist.Gamma(self.ones * self.alpha_g_phi_hyp_prior_alpha, self.ones * self.alpha_g_phi_hyp_prior_beta),
        )
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([self.n_batch, self.n_vars]).to_event(2),
        )  # (self.n_batch, self.n_vars)

        # =====================Expected expression ======================= #
        if not self.training_wo_observed:
            # expected expression
            mu = ((w_sf @ self.cell_state) * m_g + (obs2sample @ s_g_gene_add)) * detection_y_s
            alpha = obs2sample @ (self.ones / alpha_g_inverse.pow(2))
            # convert mean and overdispersion to total count and logits
            # total_count, logits = _convert_mean_disp_to_counts_logits(
            #    mu, alpha, eps=self.eps
            # )

            # =====================DATA likelihood ======================= #
            # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
            if self.dropout_p != 0:
                x_data = self.dropout(x_data)
            with obs_plate:
                pyro.sample(
                    "data_target",
                    dist.GammaPoisson(concentration=alpha, rate=alpha / mu),
                    # dist.NegativeBinomial(total_count=total_count, logits=logits),
                    obs=x_data,
                )

        # =====================Compute mRNA count from each factor in locations  ======================= #
        with obs_plate:
            mRNA = w_sf * (self.cell_state * m_g).sum(-1)
            pyro.deterministic("u_sf_mRNA_factors", mRNA)

    def compute_expected(
        self,
        samples,
        adata_manager,
        ind_x=None,
        hide_ambient=False,
        hide_cell_type=False,
    ):
        r"""Compute expected expression of each gene in each location. Useful for evaluating how well
        the model learned expression pattern of all genes in the data.
        """
        if ind_x is None:
            ind_x = np.arange(adata_manager.adata.n_obs).astype(int)
        else:
            ind_x = ind_x.astype(int)
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten()).values[ind_x, :]
        mu = np.ones((1, 1))
        if not hide_cell_type:
            mu = np.dot(samples["w_sf"][ind_x, :], self.cell_state_mat.T) * samples["m_g"]
        if not hide_ambient:
            mu = mu + np.dot(obs2sample, samples["s_g_gene_add"])
        mu = mu * samples["detection_y_s"][ind_x, :]
        alpha = np.dot(obs2sample, 1 / np.power(samples["alpha_g_inverse"], 2))

        return {"mu": mu, "alpha": alpha, "ind_x": ind_x}

    def compute_expected_per_cell_type(self, samples, adata_manager, ind_x=None):
        r"""
        Compute expected expression of each gene in each location for each cell type.

        Parameters
        ----------
        samples
            Posterior distribution summary self.samples[f"post_sample_q05}"]
            (or 'means', 'stds', 'q05', 'q95') produced by export_posterior().
        ind_x
            Location/observation indices for which to compute expected count
            (if None all locations are used).

        Returns
        -------
        dict
          dictionary with:

            1. list with expected expression counts (sparse, shape=(N locations, N genes)
               for each cell type in the same order as mod\.factor_names_;
            2. np.array with location indices
        """
        if ind_x is None:
            ind_x = np.arange(adata_manager.adata.n_obs).astype(int)
        else:
            ind_x = ind_x.astype(int)

        # fetch data
        x_data = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)[ind_x, :]
        x_data = csr_matrix(x_data)

        # compute total expected expression
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten()).values[ind_x, :]
        mu = np.dot(samples["w_sf"][ind_x, :], self.cell_state_mat.T) * samples["m_g"] + np.dot(
            obs2sample, samples["s_g_gene_add"]
        )

        # compute conditional expected expression per cell type
        mu_ct = [
            csr_matrix(
                x_data.multiply(
                    (
                        np.dot(
                            samples["w_sf"][ind_x, i, np.newaxis],
                            self.cell_state_mat.T[np.newaxis, i, :],
                        )
                        * samples["m_g"]
                    )
                    / mu
                )
            )
            for i in range(self.n_factors)
        ]

        return {"mu": mu_ct, "ind_x": ind_x}


class Cell2locationBaseModule(PyroBaseModuleClass, AutoGuideMixinModule):
    r"""
    Module class which defines AutoGuide given model. Supports multiple model architectures.

    Parameters
    ----------
    amortised
        boolean, use a Neural Network to approximate posterior distribution of location-specific (local) parameters?
    encoder_mode
        Use single encoder for all variables ("single"), one encoder per variable ("multiple")
        or a single encoder in the first step and multiple encoders in the second step ("single-multiple").
    encoder_kwargs
        arguments for Neural Network construction (scvi.nn.FCLayers)
    kwargs
        arguments for specific model class - e.g. number of genes, values of the prior distribution
    """

    def __init__(
        self,
        model,
        amortised: bool = False,
        encoder_mode: Literal["single", "multiple", "single-multiple"] = "single",
        encoder_kwargs: Optional[dict] = None,
        data_transform="log1p",
        create_autoguide_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.hist = []

        self._model = model(**kwargs)
        self._amortised = amortised
        if create_autoguide_kwargs is None:
            create_autoguide_kwargs = dict()

        self._guide = self._create_autoguide(
            model=self.model,
            amortised=self.is_amortised,
            encoder_kwargs=encoder_kwargs,
            data_transform=data_transform,
            encoder_mode=encoder_mode,
            init_loc_fn=self.init_to_value,
            n_cat_list=[kwargs["n_batch"]],
            **create_autoguide_kwargs,
        )

        self._get_fn_args_from_batch = self._model._get_fn_args_from_batch

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    @property
    def list_obs_plate_vars(self):
        """
        Create a dictionary with:

        1. "name" - the name of observation/minibatch plate;
        2. "input" - indexes of model args to provide to encoder network when using amortised inference;
        3. "sites" - dictionary with

          * keys - names of variables that belong to the observation plate
            (used to recognise and merge posterior samples for minibatch variables)
          * values - the dimensions in non-plate axis of each variable (used to
            construct output layer of encoder network when using amortised inference)
        """

        return self.model.list_obs_plate_vars()

    @property
    def is_amortised(self):
        return self._amortised

    def init_to_value(self, site):
        if getattr(self.model, "np_init_vals", None) is not None:
            init_vals = {k: getattr(self.model, f"init_val_{k}") for k in self.model.np_init_vals.keys()}
        else:
            init_vals = dict()
        return init_to_value(site=site, values=init_vals, init_fn=init_to_mean)


class PyroAggressiveTrainingPlan1(PyroTrainingPlan_scvi):
    """
    Lightning module task to train Pyro scvi-tools modules.
    Parameters
    ----------
    pyro_module
        An instance of :class:`~scvi.module.base.PyroBaseModuleClass`. This object
        should have callable `model` and `guide` attributes or methods.
    loss_fn
        A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
        If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
    optim
        A Pyro optimizer instance, e.g., :class:`~pyro.optim.Adam`. If `None`,
        defaults to :class:`pyro.optim.Adam` optimizer with a learning rate of `1e-3`.
    optim_kwargs
        Keyword arguments for **default** optimiser :class:`pyro.optim.Adam`.
    n_aggressive_epochs
        Number of epochs in aggressive optimisation of amortised variables.
    n_aggressive_steps
        Number of steps to spend optimising amortised variables before one step optimising global variables.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    """

    def __init__(
        self,
        pyro_module: PyroBaseModuleClass,
        loss_fn: Optional[pyro.infer.ELBO] = None,
        optim: Optional[pyro.optim.PyroOptim] = None,
        optim_kwargs: Optional[dict] = None,
        n_aggressive_epochs: int = 1000,
        n_aggressive_steps: int = 20,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        aggressive_vars: Union[list, None] = None,
        invert_aggressive_selection: bool = False,
    ):
        super().__init__(
            pyro_module=pyro_module,
            loss_fn=loss_fn,
            optim=optim,
            optim_kwargs=optim_kwargs,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
        )

        self.n_aggressive_epochs = n_aggressive_epochs
        self.n_aggressive_steps = n_aggressive_steps
        self.aggressive_steps_counter = 0
        self.aggressive_epochs_counter = 0
        self.mi = []
        self.n_epochs_patience = 0

        # in list not provided use amortised variables for aggressive training
        if aggressive_vars is None:
            aggressive_vars = list(self.module.list_obs_plate_vars["sites"].keys())
            aggressive_vars = aggressive_vars + [f"{i}_initial" for i in aggressive_vars]
            aggressive_vars = aggressive_vars + [f"{i}_unconstrained" for i in aggressive_vars]

        self.aggressive_vars = aggressive_vars
        self.invert_aggressive_selection = invert_aggressive_selection
        # keep frozen variables as frozen
        self.requires_grad_false_vars = [k for k, v in self.module.guide.named_parameters() if not v.requires_grad] + [
            k for k, v in self.module.model.named_parameters() if not v.requires_grad
        ]

        self.svi = pyro.infer.SVI(
            model=pyro_module.model,
            guide=pyro_module.guide,
            optim=self.optim,
            loss=self.loss_fn,
        )

    def change_requires_grad(self, aggressive_vars_status, non_aggressive_vars_status):
        for k, v in self.module.guide.named_parameters():
            if not np.any([i in k for i in self.requires_grad_false_vars]):
                k_in_vars = np.any([i in k for i in self.aggressive_vars])
                # hide variables on the list if they are not hidden
                if k_in_vars and v.requires_grad and (aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables on the list if they are hidden
                if k_in_vars and (not v.requires_grad) and (aggressive_vars_status == "expose"):
                    v.requires_grad = True

                # hide variables not on the list if they are not hidden
                if (not k_in_vars) and v.requires_grad and (non_aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables not on the list if they are hidden
                if (not k_in_vars) and (not v.requires_grad) and (non_aggressive_vars_status == "expose"):
                    v.requires_grad = True

        for k, v in self.module.model.named_parameters():
            if not np.any([i in k for i in self.requires_grad_false_vars]):
                k_in_vars = np.any([i in k for i in self.aggressive_vars])
                # hide variables on the list if they are not hidden
                if k_in_vars and v.requires_grad and (aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables on the list if they are hidden
                if k_in_vars and (not v.requires_grad) and (aggressive_vars_status == "expose"):
                    v.requires_grad = True

                # hide variables not on the list if they are not hidden
                if (not k_in_vars) and v.requires_grad and (non_aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables not on the list if they are hidden
                if (not k_in_vars) and (not v.requires_grad) and (non_aggressive_vars_status == "expose"):
                    v.requires_grad = True

    def on_train_epoch_end(self):
        self.aggressive_epochs_counter += 1

        self.change_requires_grad(
            aggressive_vars_status="expose",
            non_aggressive_vars_status="expose",
        )

        outputs = self.training_step_outputs
        elbo = 0
        n = 0
        for out in outputs:
            elbo += out["loss"]
            n += 1
        if n > 0:
            elbo /= n
        self.log("elbo_train", elbo, prog_bar=True)
        self.training_step_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        # Set KL weight if necessary.
        # Note: if applied, ELBO loss in progress bar is the effective KL annealed loss, not the true ELBO.
        if self.use_kl_weight:
            kwargs.update({"kl_weight": self.kl_weight})

        if self.aggressive_epochs_counter < self.n_aggressive_epochs:
            if self.aggressive_steps_counter < self.n_aggressive_steps:
                self.aggressive_steps_counter += 1
                # Do parameter update exclusively for amortised variables
                if self.invert_aggressive_selection:
                    self.change_requires_grad(
                        aggressive_vars_status="hide",
                        non_aggressive_vars_status="expose",
                    )
                else:
                    self.change_requires_grad(
                        aggressive_vars_status="expose",
                        non_aggressive_vars_status="hide",
                    )
                loss = torch.Tensor([self.svi.step(*args, **kwargs)])
            else:
                self.aggressive_steps_counter = 0
                # Do parameter update exclusively for non-amortised variables
                if self.invert_aggressive_selection:
                    self.change_requires_grad(
                        aggressive_vars_status="expose",
                        non_aggressive_vars_status="hide",
                    )
                else:
                    self.change_requires_grad(
                        aggressive_vars_status="hide",
                        non_aggressive_vars_status="expose",
                    )
                loss = torch.Tensor([self.svi.step(*args, **kwargs)])
        else:
            # Do parameter update for both types of variables
            self.change_requires_grad(
                aggressive_vars_status="expose",
                non_aggressive_vars_status="expose",
            )
            loss = torch.Tensor([self.svi.step(*args, **kwargs)])

        return {"loss": loss}


class PyroAggressiveTrainingPlan(PyroAggressiveTrainingPlan1):
    """
    Lightning module task to train Pyro scvi-tools modules.
    Parameters
    ----------
    pyro_module
        An instance of :class:`~scvi.module.base.PyroBaseModuleClass`. This object
        should have callable `model` and `guide` attributes or methods.
    loss_fn
        A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
        If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
    optim
        A Pyro optimizer instance, e.g., :class:`~pyro.optim.Adam`. If `None`,
        defaults to :class:`pyro.optim.Adam` optimizer with a learning rate of `1e-3`.
    optim_kwargs
        Keyword arguments for **default** optimiser :class:`pyro.optim.Adam`.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    """

    def __init__(
        self,
        scale_elbo: Union[float, None] = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if scale_elbo != 1.0:
            self.svi = pyro.infer.SVI(
                model=poutine.scale(self.module.model, scale_elbo),
                guide=poutine.scale(self.module.guide, scale_elbo),
                optim=self.optim,
                loss=self.loss_fn,
            )
        else:
            self.svi = pyro.infer.SVI(
                model=self.module.model,
                guide=self.module.guide,
                optim=self.optim,
                loss=self.loss_fn,
            )


logger = logging.getLogger(__name__)


class PyroAggressiveConvergence(Callback):
    """
    A callback to compute/apply aggressive training convergence criteria for amortised inference.
    Motivated by this paper: https://arxiv.org/pdf/1901.05534.pdf
    """

    def __init__(self, dataloader: AnnDataLoader = None, patience: int = 10, tolerance: float = 1e-4) -> None:
        super().__init__()
        self.dataloader = dataloader
        self.patience = patience
        self.tolerance = tolerance

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional[Any] = None
    ) -> None:
        """
        Compute aggressive training convergence criteria for amortised inference.
        """
        pyro_guide = pl_module.module.guide
        if hasattr(pyro_guide, "mutual_information"):
            if self.dataloader is None:
                dl = trainer.datamodule.train_dataloader()
            else:
                dl = self.dataloader
            for tensors in dl:
                tens = {k: t.to(pl_module.device) for k, t in tensors.items()}
                args, kwargs = pl_module.module._get_fn_args_from_batch(tens)
                break
            mi_ = pyro_guide.mutual_information(*args, **kwargs)
            mi_ = np.array([v for v in mi_.values()]).sum()
            pl_module.log("MI", mi_, prog_bar=True)
            if len(pl_module.mi) > 1:
                if abs(mi_ - pl_module.mi[-1]) < self.tolerance:
                    pl_module.n_epochs_patience += 1
            else:
                pl_module.n_epochs_patience = 0
            if pl_module.n_epochs_patience > self.patience:
                # stop aggressive training by setting epoch counter to max epochs
                # pl_module.aggressive_epochs_counter = pl_module.n_aggressive_epochs + 1
                logger.info('Stopped aggressive training after "{}" epochs'.format(pl_module.aggressive_epochs_counter))
            pl_module.mi.append(mi_)


def select_slide(adata, s, batch_key="sample"):
    r"""This function selects the data for one slide from the spatial anndata object.

    :param adata: Anndata object with multiple spatial experiments
    :param s: name of selected experiment
    :param batch_key: column in adata.obs listing experiment name for each location
    """

    slide = adata[adata.obs[batch_key].isin([s]), :].copy()
    s_keys = list(slide.uns["spatial"].keys())
    s_spatial = np.array(s_keys)[[s in k for k in s_keys]][0]

    slide.uns["spatial"] = {s_spatial: slide.uns["spatial"][s_spatial]}

    return slide


class Cell2location(QuantileMixin, PyroSampleMixin, PyroSviTrainMixin, PltExportMixin, BaseModelClass):
    r"""
    Cell2location model. User-end model class. See Module class for description of the model (incl. math).

    Parameters
    ----------
    adata
        spatial AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    cell_state_df
        pd.DataFrame with reference expression signatures for each gene (rows) in each cell type/population (columns).
    use_gpu
        Use the GPU?
    **model_kwargs
        Keyword args for :class:`~cell2location.models.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel`

    Examples
    --------
    TODO add example
    >>>
    """

    def __init__(
        self,
        adata: AnnData,
        cell_state_df: pd.DataFrame,
        model_class: Optional[PyroModule] = None,
        detection_mean_per_sample: bool = False,
        detection_mean_correction: float = 1.0,
        **model_kwargs,
    ):
        # in case any other model was created before that shares the same parameter names.
        clear_param_store()

        if not np.all(adata.var_names == cell_state_df.index):
            raise ValueError("adata.var_names should match cell_state_df.index, find interecting variables/genes first")

        super().__init__(adata)

        self.mi_ = []

        if model_class is None:
            model_class = LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel

        self.cell_state_df_ = cell_state_df
        self.n_factors_ = cell_state_df.shape[1]
        self.factor_names_ = cell_state_df.columns.values

        if not detection_mean_per_sample:
            # compute expected change in sensitivity (m_g in V1 or y_s in V2)
            sc_total = cell_state_df.sum(0).mean()
            sp_total = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY).sum(1).mean()
            self.detection_mean_ = (sp_total / model_kwargs.get("N_cells_per_location", 1)) / sc_total
            self.detection_mean_ = self.detection_mean_ * detection_mean_correction
            model_kwargs["detection_mean"] = self.detection_mean_
        else:
            # compute expected change in sensitivity (m_g in V1 and y_s in V2)
            sc_total = cell_state_df.sum(0).mean()
            sp_total = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY).sum(1)
            batch = self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY).flatten()
            sp_total = np.array([sp_total[batch == b].mean() for b in range(self.summary_stats["n_batch"])])
            self.detection_mean_ = (sp_total / model_kwargs.get("N_cells_per_location", 1)) / sc_total
            self.detection_mean_ = self.detection_mean_ * detection_mean_correction
            model_kwargs["detection_mean"] = self.detection_mean_.reshape((self.summary_stats["n_batch"], 1)).astype(
                "float32"
            )

        detection_alpha = model_kwargs.get("detection_alpha", None)
        if detection_alpha is not None:
            if type(detection_alpha) is dict:
                batch_mapping = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).categorical_mapping
                self.detection_alpha_ = pd.Series(detection_alpha)[batch_mapping]
                model_kwargs["detection_alpha"] = self.detection_alpha_.values.reshape(
                    (self.summary_stats["n_batch"], 1)
                ).astype("float32")

        self.module = Cell2locationBaseModule(
            model=model_class,
            n_obs=self.summary_stats["n_cells"],
            n_vars=self.summary_stats["n_vars"],
            n_factors=self.n_factors_,
            n_batch=self.summary_stats["n_batch"],
            cell_state_mat=self.cell_state_df_.values.astype("float32"),
            **model_kwargs,
        )
        self._model_summary_string = f'cell2location model with the following params: \nn_factors: {self.n_factors_} \nn_batch: {self.summary_stats["n_batch"]} '
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs).astype("int64")
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train_compiled(self, compile_mode=None, compile_dynamic=None, **kwargs):
        self.train(**kwargs, max_steps=1)
        self.module._model = torch.compile(self.module.model, mode=compile_mode, dynamic=compile_dynamic)
        self.module._guide = torch.compile(self.module.guide, mode=compile_mode, dynamic=compile_dynamic)
        self.train(**kwargs)

    def train(
        self,
        max_epochs: int = 30000,
        batch_size: int = None,
        train_size: float = 1,
        lr: float = 0.002,
        num_particles: int = 1,
        scale_elbo: float = 1.0,
        **kwargs,
    ):
        """Train the model with useful defaults

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        train_size
            Size of training set in the range [0.0, 1.0]. Use all data points in training because
            we need to estimate cell abundance at all locations.
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        kwargs
            Other arguments to scvi.model.base.PyroSviTrainMixin().train() method
        """

        kwargs["max_epochs"] = max_epochs
        kwargs["batch_size"] = batch_size
        kwargs["train_size"] = train_size
        kwargs["lr"] = lr

        if "plan_kwargs" not in kwargs.keys():
            kwargs["plan_kwargs"] = dict()
        if getattr(self.module.model, "discrete_variables", None) and (len(self.module.model.discrete_variables) > 0):
            kwargs["plan_kwargs"]["loss_fn"] = TraceEnum_ELBO(num_particles=num_particles)
        else:
            kwargs["plan_kwargs"]["loss_fn"] = Trace_ELBO(num_particles=num_particles)
        if scale_elbo != 1.0:
            if scale_elbo is None:
                scale_elbo = 1.0 / (self.summary_stats["n_cells"] * self.summary_stats["n_vars"])
            kwargs["plan_kwargs"]["scale_elbo"] = scale_elbo

        super().train(**kwargs)

    def train_aggressive(
        self,
        max_epochs: Optional[int] = 1000,
        use_gpu: Optional[Union[str, int, bool]] = None,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        train_size: float = 1,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        batch_size: int = None,
        early_stopping: bool = False,
        lr: Optional[float] = None,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            n_obs = self.adata_manager.adata.n_obs
            max_epochs = np.min([round((20000 / n_obs) * 1000), 1000])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        if lr is not None and "optim" not in plan_kwargs.keys():
            plan_kwargs.update({"optim_kwargs": {"lr": lr}})

        if batch_size is None:
            # use data splitter which moves data to GPU once
            data_splitter = DeviceBackedDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
                accelerator=accelerator,
                device=device,
            )
        else:
            data_splitter = self._data_splitter_cls(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                shuffle_set_split=shuffle_set_split,
                batch_size=batch_size,
            )
        training_plan = PyroAggressiveTrainingPlan(pyro_module=self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]

        if "callbacks" not in trainer_kwargs.keys():
            trainer_kwargs["callbacks"] = []
        trainer_kwargs["callbacks"].append(PyroJitGuideWarmup())
        trainer_kwargs["callbacks"].append(PyroAggressiveConvergence())

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            accelerator=accelerator,
            devices=device,
            **trainer_kwargs,
        )
        res = runner()
        self.mi_ = self.mi_ + training_plan.mi
        return res

    def export_posterior(
        self,
        adata,
        sample_kwargs: Optional[dict] = None,
        export_slot: str = "mod",
        add_to_obsm: list = ["means", "stds", "q05", "q95"],
        use_quantiles: bool = False,
    ):
        """
        Summarise posterior distribution and export results (cell abundance) to anndata object:

        1. adata.obsm: Estimated cell abundance as pd.DataFrames for each posterior distribution summary `add_to_obsm`,
            posterior mean, sd, 5% and 95% quantiles (['means', 'stds', 'q05', 'q95']).
            If export to adata.obsm fails with error, results are saved to adata.obs instead.
        2. adata.uns: Posterior of all parameters, model name, date,
            cell type names ('factor_names'), obs and var names.

        Parameters
        ----------
        adata
            anndata object where results should be saved
        sample_kwargs
            arguments for self.sample_posterior (generating and summarising posterior samples), namely:
                num_samples - number of samples to use (Default = 1000).
                batch_size - data batch size (keep low enough to fit on GPU, default 2048).
                use_gpu - use gpu for generating samples?
        export_slot
            adata.uns slot where to export results
        add_to_obsm
            posterior distribution summary to export in adata.obsm (['means', 'stds', 'q05', 'q95']).
        use_quantiles
            compute quantiles directly (True, more memory efficient) or use samples (False, default).
            If True, means and stds cannot be computed so are not exported and returned.
        Returns
        -------

        """

        sample_kwargs = sample_kwargs if isinstance(sample_kwargs, dict) else dict()

        # get posterior distribution summary
        if use_quantiles:
            add_to_obsm = [i for i in add_to_obsm if (i not in ["means", "stds"]) and ("q" in i)]
            if len(add_to_obsm) == 0:
                raise ValueError("No quantiles to export - please add add_to_obsm=['q05', 'q50', 'q95'].")
            self.samples = dict()
            for i in add_to_obsm:
                q = float(f"0.{i[1:]}")
                self.samples[f"post_sample_{i}"] = self.posterior_quantile(q=q, **sample_kwargs)
        else:
            # generate samples from posterior distributions for all parameters
            # and compute mean, 5%/95% quantiles and standard deviation
            self.samples = self.sample_posterior(**sample_kwargs)

        # export posterior distribution summary for all parameters and
        # annotation (model, date, var, obs and cell type names) to anndata object
        adata.uns[export_slot] = self._export2adata(self.samples)

        # add estimated cell abundance as dataframe to obsm in anndata
        # first convert np.arrays to pd.DataFrames with cell type and observation names
        # data frames contain mean, 5%/95% quantiles and standard deviation, denoted by a prefix
        for k in add_to_obsm:
            sample_df = self.sample2df_obs(
                self.samples,
                site_name="w_sf",
                summary_name=k,
                name_prefix="cell_abundance",
            )
            try:
                adata.obsm[f"{k}_cell_abundance_w_sf"] = sample_df.loc[adata.obs.index, :]
            except ValueError:
                # Catching weird error with obsm: `ValueError: value.index does not match parent’s axis 1 names`
                adata.obs[sample_df.columns] = sample_df.loc[adata.obs.index, :]

        return adata

    def plot_spatial_QC_across_batches(self):
        """QC plot: compare total RNA count with estimated total cell abundance and detection sensitivity."""

        adata = self.adata

        # get batch key and the list of samples
        batch_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).original_key
        samples = adata.obs[batch_key].unique()

        # figure out plot shape
        ncol = len(samples)
        nrow = 3
        fig, axs = plt.subplots(nrow, ncol, figsize=(1 + 4 * ncol, 1 + 4 * nrow))
        if ncol == 1:
            axs = axs.reshape((nrow, 1))

        # compute total counts
        # find data slot
        x_dict = self.adata_manager.data_registry[REGISTRY_KEYS.X_KEY]
        if x_dict["attr_name"] == "X":
            use_raw = False
        else:
            use_raw = True
        if x_dict["attr_name"] == "layers":
            layer = x_dict["attr_key"]
        else:
            layer = None

        # get data
        if layer is not None:
            x = adata.layers[layer]
        else:
            if not use_raw:
                x = adata.X
            else:
                x = adata.raw.X
        # compute total counts per location
        cell_type = "total RNA counts"
        adata.obs[cell_type] = np.array(x.sum(1)).flatten()

        # figure out colour map scaling
        vmax = np.quantile(adata.obs[cell_type].values, 0.992)
        # plot, iterating across samples
        for i, s in enumerate(samples):
            sp_data_s = select_slide(adata, s, batch_key=batch_key)
            scanpy.pl.spatial(
                sp_data_s,
                cmap="magma",
                color=cell_type,
                size=1.3,
                img_key="hires",
                alpha_img=1,
                vmin=0,
                vmax=vmax,
                ax=axs[0, i],
                show=False,
            )
            axs[0, i].title.set_text(cell_type + "\n" + s)

        cell_type = "Total cell abundance (sum_f w_sf)"
        adata.obs[cell_type] = adata.uns["mod"]["post_sample_means"]["w_sf"].sum(1).flatten()
        # figure out colour map scaling
        vmax = np.quantile(adata.obs[cell_type].values, 0.992)
        # plot, iterating across samples
        for i, s in enumerate(samples):
            sp_data_s = select_slide(adata, s, batch_key=batch_key)
            scanpy.pl.spatial(
                sp_data_s,
                cmap="magma",
                color=cell_type,
                size=1.3,
                img_key="hires",
                alpha_img=1,
                vmin=0,
                vmax=vmax,
                ax=axs[1, i],
                show=False,
            )
            axs[1, i].title.set_text(cell_type + "\n" + s)

        cell_type = "RNA detection sensitivity (y_s)"
        adata.obs[cell_type] = adata.uns["mod"]["post_sample_q05"]["detection_y_s"]
        # figure out colour map scaling
        vmax = np.quantile(adata.obs[cell_type].values, 0.992)
        # plot, iterating across samples
        for i, s in enumerate(samples):
            sp_data_s = select_slide(adata, s, batch_key=batch_key)
            scanpy.pl.spatial(
                sp_data_s,
                cmap="magma",
                color=cell_type,
                size=1.3,
                img_key="hires",
                alpha_img=1,
                vmin=0,
                vmax=vmax,
                ax=axs[2, i],
                show=False,
            )
            axs[2, i].title.set_text(cell_type + "\n" + s)

        fig.tight_layout(pad=0.5)

        return fig
