# Adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/checkpoint.py
# and https://github.com/google-research/vision_transformer/blob/main/vit_jax/input_pipeline.py

import collections
from collections import abc
import flax
from flax.training import checkpoints
import re
from packaging import version
import numpy as np
import jax.numpy as jnp
import scipy

import glob
import os
import sys

from absl import logging
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms


def recover_tree(keys, values):
    """Recovers a tree as a nested dict from flat names and values.
    This function is useful to analyze checkpoints that are without need to access
    the exact source code of the experiment. In particular, it can be used to
    extract an reuse various subtrees of the scheckpoint, e.g. subtree of
    parameters.
    Args:
      keys: a list of keys, where '/' is used as separator between nodes.
      values: a list of leaf values.
    Returns:
      A nested tree-like dict.
    """
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def _flatten_dict(d, parent_key="", sep="/"):
    """Flattens a dictionary, keeping empty leaves."""
    items = []
    for k, v in d.items():
        path = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.Mapping):
            items.extend(_flatten_dict(v, path, sep=sep).items())
        else:
            items.append((path, v))

    # Keeps the empty dict if it was set explicitly.
    if parent_key and not d:
        items.append((parent_key, {}))

    return dict(items)


def load(path):
    """Loads params from a checkpoint previously stored with `save()`."""
    with open(path, "rb") as f:
        ckpt_dict = np.load(f, allow_pickle=False)
        keys, values = zip(*list(ckpt_dict.items()))
    params = checkpoints.convert_pre_linen(recover_tree(keys, values))
    if isinstance(params, flax.core.FrozenDict):
        params = params.unfreeze()
    if version.parse(flax.__version__) >= version.parse("0.3.6"):
        params = _fix_groupnorm(params)
    return params


def _fix_groupnorm(params):
    # See https://github.com/google/flax/issues/1721
    regex = re.compile(r"gn(\d+|_root|_proj)$")

    def fix_gn(args):
        path, array = args
        if len(path) > 1 and regex.match(path[-2]) and path[-1] in ("bias", "scale"):
            array = array.squeeze()
        return (path, array)

    return flax.traverse_util.unflatten_dict(
        dict(map(fix_gn, flax.traverse_util.flatten_dict(params).items()))
    )


def inspect_params(*, params, expected, fail_if_extra=True, fail_if_missing=True):
    """Inspects whether the params are consistent with the expected keys."""
    params_flat = _flatten_dict(params)
    expected_flat = _flatten_dict(expected)
    missing_keys = expected_flat.keys() - params_flat.keys()
    extra_keys = params_flat.keys() - expected_flat.keys()

    # Adds back empty dict explicitly, to support layers without weights.
    # Context: FLAX ignores empty dict during serialization.
    empty_keys = set()
    for k in missing_keys:
        if isinstance(expected_flat[k], dict) and not expected_flat[k]:
            params[k] = {}
            empty_keys.add(k)
    missing_keys -= empty_keys

    if empty_keys:
        print("Inspect recovered empty keys:\n%s", empty_keys)
    if missing_keys:
        print("Inspect missing keys:\n%s", missing_keys)
    if extra_keys:
        print("Inspect extra keys:\n%s", extra_keys)

    if (missing_keys and fail_if_missing) or (extra_keys and fail_if_extra):
        raise ValueError(
            f"Missing params from checkpoint: {missing_keys}.\n"
            f"Extra params in checkpoint: {extra_keys}.\n"
            f"Restored params from checkpoint: {params_flat.keys()}.\n"
            f"Expected params from code: {expected_flat.keys()}."
        )
    return params


def load_pretrained(*, pretrained_path, init_params):
    """Loads/converts a pretrained checkpoint for fine tuning.
    Args:
      pretrained_path: File pointing to pretrained checkpoint.
      init_params: Parameters from model. Will be used for the head of the model
        and to verify that the model is compatible with the stored checkpoint.
    Returns:
      Parameters like `init_params`, but loaded with pretrained weights from
      `pretrained_path` and adapted accordingly.
    """

    restored_params = inspect_params(
        params=load(pretrained_path),
        expected=init_params,
        fail_if_extra=False,
        fail_if_missing=False,
    )

    # The following allows implementing fine-tuning head variants depending on the
    # value of `representation_size` in the fine-tuning job:
    # - `None` : drop the whole head and attach a nn.Linear.
    # - same number as in pre-training means : keep the head but reset the last
    #    layer (logits) for the new task.
    if "pre_logits" in restored_params:
        print("load_pretrained: drop-head variant")
        restored_params["pre_logits"] = {}
    restored_params["head"]["kernel"] = init_params["head"]["kernel"]
    restored_params["head"]["bias"] = init_params["head"]["bias"]

    if "posembed_input" in restored_params.get("Transformer", {}):
        # Rescale the grid of position embeddings. Param shape is (1,N,1024)
        posemb = restored_params["Transformer"]["posembed_input"]["pos_embedding"]
        posemb_new = init_params["Transformer"]["posembed_input"]["pos_embedding"]
        if posemb.shape != posemb_new.shape:
            print(
                "load_pretrained: resized variant: %s to %s",
                posemb.shape,
                posemb_new.shape,
            )
            ntok_new = posemb_new.shape[1]

            posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
            ntok_new -= 1

            gs_old = int(np.sqrt(len(posemb_grid)))
            gs_new = int(np.sqrt(ntok_new))
            print("load_pretrained: grid-size from %s to %s", gs_old, gs_new)
            posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

            zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
            posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            posemb = jnp.array(np.concatenate([posemb_tok, posemb_grid], axis=1))
            restored_params["Transformer"]["posembed_input"]["pos_embedding"] = posemb

    if version.parse(flax.__version__) >= version.parse("0.3.6"):
        restored_params = _fix_groupnorm(restored_params)

    return flax.core.freeze(restored_params)


def get_cifar10():
    """Download (if necessary) and return the CIFAR10 dataset."""
    # The following is a workaround for this bug: https://github.com/pytorch/vision/issues/5039
    if sys.platform == "win32":
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
    # Magic constants taken from: https://docs.ffcv.io/ffcv_examples/cifar10.html
    mean = torch.tensor([125.307, 122.961, 113.8575]) / 255
    std = torch.tensor([51.5865, 50.847, 51.255]) / 255
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
            transforms.Resize(224),
        ]
    )
    cifar_train = torchvision.datasets.CIFAR10(
        "cifar10_train", transform=transform, download=True, train=True
    )
    cifar_test = torchvision.datasets.CIFAR10(
        "cifar10_train", transform=transform, download=True, train=False
    )
    return (cifar_train, cifar_test)


# NumpyLoader is adapted from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)
