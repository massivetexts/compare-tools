import yaml
import os

config = None
# Looks for a file named "local.yaml" in the current directory or higher in the directory tree.

for path in ['local.yaml', '../local.yaml', '../../local.yaml', '../../../local.yaml']:
    try:
        config = yaml.safe_load(open(path))
    except FileNotFoundError:
        pass

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

def init_htid_args(config):
    ''' Using arguments from config, initialize HathiMeta and Vector_file objects
    and return a dict that can be passed easily to HTID, expanded with **kwargs '''
    from .hathimeta import HathiMeta
    from SRP import Vector_file

    metastore = HathiMeta(config['metadb_path'])
    data_path_keys = [name[:-10] for name in config.keys() if name.endswith('_data_path')]
    
    args = dict(ef_root=config['parquet_root'],
                ef_chunk_root=config['parquet_chunked_root'],
                ef_parser='parquet',
                hathimeta=metastore,
                vecfiles=[]
               )
    
    for key in data_path_keys:
        args['vecfiles'].append((key, Vector_file(config[key+'_data_path'], mode='r')))
        
    return args