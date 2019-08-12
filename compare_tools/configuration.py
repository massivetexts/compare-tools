import yaml
import os

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
    SRP_path = config['SRP_data_path']
except:
    pass

try:
    pca_path = config['pca_path']
except:
    pass
