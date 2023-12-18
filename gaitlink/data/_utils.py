from functools import lru_cache
from typing import Callable, Optional, TypeVar, Union

from joblib import Memory
from tpcp._hash import custom_hash

T = TypeVar("T")


_GLOBAL_CACHE: dict[str, Callable] = {}


def staggered_cache(
    function, joblib_memory: Memory = Memory(None), lru_cache_maxsize: Union[Optional[int], bool] = None
):
    """A staggered cache that first uses a joblib memory cache and then a lru cache."""
    paras_hash = custom_hash((function, joblib_memory, lru_cache_maxsize))
    if paras_hash in _GLOBAL_CACHE:
        return _GLOBAL_CACHE[paras_hash]

    if lru_cache_maxsize is False:
        final_cached = joblib_memory.cache(function)
    else:

        def inner_cached(*args, **kwargs):
            return joblib_memory.cache(function)(*args, **kwargs)

        final_cached = lru_cache(lru_cache_maxsize)(inner_cached)

    _GLOBAL_CACHE[paras_hash] = final_cached

    return final_cached


staggered_cache.__cache__ = _GLOBAL_CACHE
