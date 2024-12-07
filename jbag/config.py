import os
import re
import tomllib
from typing import Mapping, Union, LiteralString


class Config(Mapping):
    def __init__(self, config: Mapping):
        self._config = config

    def __getattr__(self, name):
        try:
            value = self._config[name]
        except KeyError:
            raise AttributeError(f'The config has no key {name}') from None
        if isinstance(value, Mapping):
            return Config(value)
        return value

    def __getitem__(self, key):
        try:
            value = self._config[key]
        except KeyError:
            raise KeyError(f'The config has no key {key}') from None
        if isinstance(value, Mapping):
            return Config(value)
        return value

    def as_primitive(self) -> dict:
        return self._config

    def __dir__(self) -> list:
        return list(self._config.keys())

    def __repr__(self) -> str:
        return repr(self._config)

    def __str__(self) -> str:
        return f"configuration keys: {dir(self)}"

    def __len__(self):
        return len(self._config)

    def __iter__(self):
        return iter(self._config)


def load_config(file: Union[str, LiteralString]):
    assert os.path.isfile(file), f'{file} does not exist or is not a file!'

    with open(file, 'rb') as f:
        config = tomllib.load(f)
    refine_nodes(config, config)
    config = Config(config)
    return config


def refine_nodes(node, root):
    for k, v in node.items():
        if isinstance(v, str):
            replace_str(k, node, root)
        elif isinstance(v, dict):
            refine_nodes(v, root)
        elif isinstance(v, list):
            for i, inner_v in enumerate(v):
                if isinstance(inner_v, str):
                    replace_str(k, node, root, list_index=i)
                elif isinstance(inner_v, dict):
                    refine_nodes(inner_v, root)


def replace_str(key, node, root, list_index=None):
    """
    If `node[key]` is list type, set list_index to the index of current element of `node[key]`.

    Args:
        key (str):
        node (mapping):
        root (mapping):
        list_index (int or None, optional, default=None):

    Returns:

    """
    reference_patten = r'(\$\{.+?\})'
    reference_key_patten = r'\$\{(.+)\}'
    if list_index is not None:
        value = node[key][list_index]
    else:
        value = node[key]
    match = re.findall(reference_patten, value)
    if not match:
        return
    for each in match:
        replace_key = re.search(reference_key_patten, each).group(1)
        replace_node, replace_key = get_node_key(replace_key, node, root)
        if isinstance(replace_node[replace_key], str):
            replace_str(replace_key, replace_node, root)
        if list_index is not None:
            node[key][list_index] = node[key][list_index].replace(each, root[replace_key])
        else:
            node[key] = node[key].replace(each, str(replace_node[replace_key]))


def get_node_key(key, node, root):
    """
    Get the inner node and key.

    Args:
        key (str):
        node (dict):
        root (mapping):

    Returns:

    """
    node_hierarchy = key.split('.')
    # first, search on current node
    # second, search on root if search failed on current node

    node, unpacked_key = search_node(node_hierarchy, node)
    if node and unpacked_key:
        return node, unpacked_key

    node = root
    node, unpacked_key = search_node(node_hierarchy, node)
    if node and unpacked_key:
        return node, unpacked_key
    raise ValueError(f'The config has no key {key}')


def search_node(node_hierarchy, node):
    for i, key in enumerate(node_hierarchy):
        if key in node:
            if i == len(node_hierarchy) - 1:
                return node, node_hierarchy[-1]
            else:
                node = node[key]
    return None, None
