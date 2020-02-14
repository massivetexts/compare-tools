import yaml
from pathlib import Path
import htrc_features
from htrc_features import caching, Volume

# You can change this here to get a different resolver: but it's better to simply edit a file at ~/.htrc-config.yaml

default = """                                                                                                                                                    
resolver:                                                                                                                                                        
  -                                                                                                                                                              
    id_resolver: pairtree                                                                                                                                        
    dir: /drobo/feature-counts                                                                                                                                   
    format: json                                                                                                                                                 
    compression: bz2                                                                                                                                             
"""

try:
    config = yaml.safe_load(Path("~/.htrc-config.yaml").expanduser().open())
except FileNotFoundError:
    raise
    config = yaml.safe_load(default)

resolver = config['resolver']

def resolver_factory(id_resolver, **kwargs):
    return htrc_features.resolvers.resolver_nicknames[id_resolver](**kwargs)

def combine_resolvers(l):
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

my_resolver = combine_resolvers(resolver)

if __name__ == "__main__":
    print(Volume(id="mdp.39015012434786", id_resolver = my_resolver).tokenlist(pos=False, section="default"))



