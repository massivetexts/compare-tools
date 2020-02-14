import yaml
import os
from pathlib import Path

config = {}
# Looks for a file named "local.yaml" in the current directory or higher in the directory tree.

for depth in range(7):
    for fname in ['local.yaml', '.htrc-config.yaml']:
        path = Path('../' * depth, fname)
        if path.exists():
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
    raise KeyError("""You must define a resolver for your system somewhere in the search path,
which is files called 'local.yaml' or '.htrc-config.yaml'.

    Here's what mine looks like on my laptop; first it looks in a flat folder of gzip-compression parquet; then in a pairtree; and it fails those, pulls from the web.


resolver:
  -
    dir: /home/bschmidt/hathi-parquet
    format: parquet
    compression: gzip
    index: false
    indexed: false
    cache: true
    token_kwargs:
      pos: false
      section: body
      drop_section: true
    id_resolver: local
  - 
    dir: /home/bschmidt/hathi-ef
    format: json
    cache: true
    compression: bz2
    id_resolver: pairtree
  - 
    id_resolver: http

""")

try:
    resolver_args = config['chunk_resolver']
    chunk_resolver = combine_resolvers(resolver_args)    
except KeyError:
    # The chunk one can just be the regular resolver.
    chunk_resolver = my_resolver
    
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
        args['vecfiles'].append((key, Vector_file(config[key+'_data_path'], mode='r')))
        
    return args
