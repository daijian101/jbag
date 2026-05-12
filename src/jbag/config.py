import os
import re
import tomllib
from typing import Mapping, Any


class Config(Mapping[str, Any]):
    _REFERENCE_PATTERN = re.compile(r'\$\{([^}]+)}')

    def __init__(self, data: Mapping[str, Any], root: Mapping[str, Any] = None):
        self._data = data
        self._root = root or data
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key):
        return self._get_value(key)

    def __getattr__(self, item):
        if item in self._data:
            return self._get_value(item)
        raise AttributeError(f'{item} not found.')

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f'Config({self._data!r})'

    def __str__(self):
        return f'Config keys: {', '.join(self._data.keys())}'

    def __dir__(self):
        return list(self._data.keys())

    def as_dict(self) -> dict:
        resolved = {}
        for key in self._data.keys():
            val = self._get_value(key)
            resolved[key] = val.as_dict() if isinstance(val, Config) else val
        return resolved

    def _get_value(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]

        raw = self._data[key]
        if isinstance(raw, Mapping):
            value = Config(raw, self._root)
        elif isinstance(raw, list):
            value = [self._resolve_list_item(v) for v in raw]
        elif isinstance(raw, str):
            value = self._resolve_string(raw)
        else:
            value = raw

        self._cache[key] = value
        return value

    def _resolve_list_item(self, item: Any) -> Any:
        if isinstance(item, str):
            return self._resolve_string(item)
        return item

    def _resolve_string(self, text: str) -> str:
        matches = self._REFERENCE_PATTERN.findall(text)
        if not matches:
            return text
        for ref in matches:
            if ref.startswith('env:'):
                # Inject environment variable
                env_key = ref[len('env:'):]
                env_val = os.getenv(env_key)
                if env_val is None:
                    raise KeyError(f'Environment variable "{env_key}" not found.')
                replacement = env_val
            else:
                # Inject config reference
                data, key = self._locate(ref)
                replacement = data[key]
                if isinstance(replacement, str):
                    replacement = self._resolve_string(replacement)
                elif isinstance(replacement, (Mapping, list)):
                    raise ValueError(f'Unsupported data type of "{type(replacement)}" for injection.')
            text = text.replace(f'${{{ref}}}', str(replacement))
        return text

    def _locate(self, path: str):
        key_chain = path.split('.')

        data, key = Config._traverse(self._data, key_chain)
        if data and key:
            return data, key

        data, key = Config._traverse(self._root, key_chain)
        if data and key:
            return data, key
        raise KeyError(f'{path} not found.')

    @staticmethod
    def _traverse(data: Mapping[str, Any], key_chain: list[str]):
        """
        Traverse and get key-word based on the chain of keys
        Args:
            data (Mapping):
            key_chain (list): Chain of keys

        Returns:
            Return the last layer mapping and key if matched or None.
        """
        for i, key in enumerate(key_chain):
            if not isinstance(data, Mapping) or key not in data:
                return None, None
            if i == len(key_chain) - 1:
                return data, key
            data = data[key]
        return None, None


def load_toml_config(config_file: str):
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f'Config file {config_file} not found.')

    with open(config_file, 'rb') as f:
        data = tomllib.load(f)
    return Config(data)
