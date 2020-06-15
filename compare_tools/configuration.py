import yaml
import os
from pathlib import Path

config = {}

# Looks for a file named "local.yaml" in the current directory or higher in the directory tree,
# or at '.htrc_config.yaml'.

used = []
pathset = []
    
for depth in range(7):
    for fname in ['local.yaml', '.htrc-config.yaml']:
        pathset.append(Path('../' * depth, fname))
pathset.append(Path("~/.htrc-config.yaml").expanduser())
for path in pathset:
    if path.exists():
        used.append(path)
        config.update(yaml.safe_load(open(path)))

# Some of these can be imported directly, if they're there.
try:
    glove_data_path = config["glove_data_path"]
except:
    pass

try:
    meta_path = config["meta_path"]
except:
    pass

try:
    rsync_root = config['rsync_root']
except:
    pass

try:
    parquet_root = config['parquet_root']
except:
    pass

try:
    SRP_path = config['SRP_data_path']
except:
    pass

try:
    pca_path = config['pca_path']
except:
    pass

from .resolver import combine_resolvers

try:
    resolver_args = config['resolver']
    my_resolver = combine_resolvers(resolver_args)
except:
    my_resolver = None

try:
    resolver_args = config['chunk_resolver']
    chunk_resolver = combine_resolvers(resolver_args)    
except KeyError:
    # The chunk one can just be the regular resolver.
    chunk_resolver = my_resolver

def wem_loader(model_name):
    ''' Load Gensim-style WEM model. e.g. 'glove-wiki-gigaword-100' '''
    import gensim.downloader as api
    api.BASE_DIR, api.base_dir = config['gensim_data_path'], config['gensim_data_path']
    return api.load(model_name)
    
def init_htid_args(config):
    ''' Using arguments from config, initialize HathiMeta and Vector_file objects
    and return a dict that can be passed easily to HTID, expanded with **kwargs '''
    from .hathimeta import HathiMeta

    from SRP import Vector_file

    metastore = HathiMeta(config['metadb_path'])
    data_path_keys = [name[:-10] for name in config.keys() if name.endswith('_data_path')]
    
    args = dict(id_resolver=my_resolver,
                chunk_resolver = chunk_resolver,
                hathimeta=metastore,
                vecfiles=[]
               )
    
    for key in data_path_keys:
        args['vecfiles'].append((key, Vector_file(config[key+'_data_path'], mode='r', offset_cache=True)))
        
    return args
