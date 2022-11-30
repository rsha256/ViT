# Adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/checkpoint.py

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
    if '/' not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split('/', 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def _flatten_dict(d, parent_key='', sep='/'):
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


def inspect_params(*,
                   params,
                   expected,
                   fail_if_extra=True,
                   fail_if_missing=True):
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
    logging.warning('Inspect recovered empty keys:\n%s', empty_keys)
  if missing_keys:
    logging.info('Inspect missing keys:\n%s', missing_keys)
  if extra_keys:
    logging.info('Inspect extra keys:\n%s', extra_keys)

  if (missing_keys and fail_if_missing) or (extra_keys and fail_if_extra):
    raise ValueError(f'Missing params from checkpoint: {missing_keys}.\n'
                     f'Extra params in checkpoint: {extra_keys}.\n'
                     f'Restored params from checkpoint: {params_flat.keys()}.\n'
                     f'Expected params from code: {expected_flat.keys()}.')
  return params


def load_pretrained(*, pretrained_path, init_params, model_config):
  """Loads/converts a pretrained checkpoint for fine tuning.
  Args:
    pretrained_path: File pointing to pretrained checkpoint.
    init_params: Parameters from model. Will be used for the head of the model
      and to verify that the model is compatible with the stored checkpoint.
    model_config: Configuration of the model. Will be used to configure the head
      and rescale the position embeddings.
  Returns:
    Parameters like `init_params`, but loaded with pretrained weights from
    `pretrained_path` and adapted accordingly.
  """

  restored_params = inspect_params(
      params=load(pretrained_path),
      expected=init_params,
      fail_if_extra=False,
      fail_if_missing=False)

  # The following allows implementing fine-tuning head variants depending on the
  # value of `representation_size` in the fine-tuning job:
  # - `None` : drop the whole head and attach a nn.Linear.
  # - same number as in pre-training means : keep the head but reset the last
  #    layer (logits) for the new task.
  if model_config.get('representation_size') is None:
    if 'pre_logits' in restored_params:
      logging.info('load_pretrained: drop-head variant')
      restored_params['pre_logits'] = {}
  restored_params['head']['kernel'] = init_params['head']['kernel']
  restored_params['head']['bias'] = init_params['head']['bias']

  if 'posembed_input' in restored_params.get('Transformer', {}):
    # Rescale the grid of position embeddings. Param shape is (1,N,1024)
    posemb = restored_params['Transformer']['posembed_input']['pos_embedding']
    posemb_new = init_params['Transformer']['posembed_input']['pos_embedding']
    if posemb.shape != posemb_new.shape:
      logging.info('load_pretrained: resized variant: %s to %s', posemb.shape,
                   posemb_new.shape)
      ntok_new = posemb_new.shape[1]

      if model_config.classifier == 'token':
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
      else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

      gs_old = int(np.sqrt(len(posemb_grid)))
      gs_new = int(np.sqrt(ntok_new))
      logging.info('load_pretrained: grid-size from %s to %s', gs_old, gs_new)
      posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

      zoom = (gs_new / gs_old, gs_new / gs_old, 1)
      posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
      posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
      posemb = jnp.array(np.concatenate([posemb_tok, posemb_grid], axis=1))
      restored_params['Transformer']['posembed_input']['pos_embedding'] = posemb

  if version.parse(flax.__version__) >= version.parse('0.3.6'):
    restored_params = _fix_groupnorm(restored_params)

  return flax.core.freeze(restored_params)