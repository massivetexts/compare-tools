import os
import pandas as pd
from htrc_features import FeatureReader, Volume
from htrc_features.feature_reader import group_tokenlist
from htrc_features.utils import id_to_rsync
import pandas as pd
import numpy as np
import json
import urllib
import altair as alt
import warnings

try:
    from .configuration import parquet_root
except:
    warnings.warn("No rsync root found")
    parquet_root = None

alt.data_transformers.enable('json')

def split_mtid(mtid):
    htid, seq = mtid.split('-')
    seq = int(seq)
    return htid, seq
    
def join_mtid(htid, seq):
    return "%s-%04.f" % (htid, seq)

def htid_ize(mtid):
    """
    Extract the MTID from an HTID.
    """
    return mtid.split("-")[0]


class MTVolume():
    def __init__(self, htid, corpus):
        """
        Initialized with an HTID and with a corpus that contains it.
        """
        self.htid = htid
        self.corpus = corpus
        self.parts, self.vectorset = corpus.htid_matrix([htid])
        
    def brute_cosine(self):
        distances = np.dot(self.vectorset, self.corpus.matrix.T)
        return distances

    def neighbors(self, comps_per_item = 10, minimum_matches = 5):
        ds = self.brute_cosine()
        partitioned = np.argpartition(ds, -comps_per_item, axis=1)[...,-comps_per_item:]
        neighbor_names = np.array(self.corpus.ids)[partitioned]

        arr = map(htid_ize, np.array(neighbor_names.flatten()))
        from collections import Counter
        commons = Counter(arr).most_common(10)
        candidates = [c for c in commons if c[1] > minimum_matches and c[0] != self.htid]
        return candidates


class HTID(object):
    """
    Initialize a volume directly from the reader files.

    Essentially a wrapper around the feature reader without any iteration.
    """
    def __init__(self, htid, rsync_root=None, parquet_root=None):
        self.htid = htid
        self.reader = None
        self._volume = None
        self._tokensets = None
        self.rsync_root = rsync_root
        self.parquet_root =  parquet_root
        assert self.rsync_root or self.parquet_root
      
    @property
    def _loc(self):
        if self.rsync_root:
            loc = id_to_rsync(self.htid)
            return os.path.join(self.rsync_root, loc)
        elif self.parquet_root: 
            loc = id_to_rsync(self.htid).replace('.json.bz2', '')
            return os.path.join(self.parquet_root, loc)
    @property
    def volume(self):
        if not self._volume:
            self._volume = Volume(self._loc, parser=('parquet' if self.parquet_root else 'json'))
        return self._volume

    @property
    def page_counts(self):
        """
        Return a pandas frame with page, token, and count information.

        Searches in two places before returning: first, a local cache:
        second, an on-disk cache in the pairtree location ending with parquet.
        """
        try:
            return self._page_counts
        except AttributeError:
            pass
        
        return self.volume.tokenlist(case=False, pos=False)
    
    @property
    def tokensets(self):
        if self._tokensets is not None:
            return self._tokensets
        self._tokensets = self.page_counts.reset_index().groupby(self.volume._pagecolname).apply(lambda x: set(x['lowercase']))
        return self._tokensets
    
    def _repr_html_(self):
        return self.volume._repr_html_()
        
    
def get_json_meta(htid, parquet_root):
    ''' Quickly read a pairtree-organized metadata file that accompanies 
    the Parquet Feature Reader export.'''
    from htrc_features import utils
    import ujson as json
    path = parquet_root + utils.id_to_rsync(htid).replace('json.bz2', 'meta.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
