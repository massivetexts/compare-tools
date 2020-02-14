import yaml
from pathlib import Path
import htrc_features
from htrc_features import caching, Volume

def resolver_factory(id_resolver, **kwargs):
    return htrc_features.resolvers.resolver_nicknames[id_resolver](**kwargs)

def combine_resolvers(l):
    if len(l) == 1:
        return resolver_factor(**l[0])
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



