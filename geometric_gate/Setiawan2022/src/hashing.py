#!/usr/bin/env python3
""" Hashing utilities to use in object hashing definitions
"""
from typing import Any, Iterable


def hash_iterable(target: Iterable) -> int:
    if isinstance(target, str):
        return hash(target)
    hash_list = []
    for x in target:
        if isinstance(x, Iterable):
            hash_list.append(hash_iterable(x))
        else:
            hash_list.append(hash(x))
    hash_tuple = tuple(hash_list)
    return hash(hash_tuple)

def hash_dict(target: dict[str, Any]) -> int:
    hash_list = []
    for key, value in target.items():
        if isinstance(value, dict):
            hash_list.append(hash((key, hash_dict(value))))
        elif isinstance(value, Iterable):
            hash_list.append(hash((key, hash_iterable(value))))
        else:
            hash_list.append(hash((key,value)))
    hash_tuple = tuple(hash_list)
    return hash(hash_tuple)
