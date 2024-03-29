import yaml
from pathlib import Path
import htrc_features
from htrc_features import caching, Volume, resolvers
from htrc_features.utils import _id_encode
import os

def resolver_factory(id_resolver, **kwargs):
    if '.' in id_resolver:
        # Assume a class import
        parts = id_resolver.split('.')
        module_path, class_name = ".".join(parts[:-1]), parts[-1]
        mod = __import__(module_path, fromlist=[class_name])
        resolver_class = getattr(mod, class_name)
    elif id_resolver in htrc_features.resolvers.resolver_nicknames:
        resolver_class = htrc_features.resolvers.resolver_nicknames[id_resolver]
    else:
        raise Exception("No resolver found matching your config")
    return resolver_class(**kwargs)

def combine_resolvers(l):
    if type(l) is not list:
        l = [l]
    if len(l) == 1:
        return resolver_factory(**l[0])
    assert(len(l) >= 2)
    first = l[0]
    rest = l[1:]
    if len(rest) > 1:
        second_choice = combine_resolvers(rest)
    else:
        second_choice = resolver_factory(**rest[0])
    with_fallback = caching.make_fallback_resolver(first['id_resolver'], second_choice, cache = first.get("cache", True))
    del first['id_resolver']
    return with_fallback(**first)
