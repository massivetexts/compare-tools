import os
from htrc_features import FeatureReader, Volume
from htrc_features.feature_reader import group_tokenlist
from htrc_features.utils import id_to_rsync
import pandas as pd
import numpy as np
import json
import urllib
import warnings

try:
    from .configuration import rsync_root
except:
    warnings.warn("No rsync root found")
    parquet_root = None

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
    A wrapper around various sources of volume-level metadata and data. This
    is for convenience - it contains little logic itself.
    
    Most arguments are optional - if you try to call information without the
    necessary source, it will warn you.
    """
    def __init__(self, htid, ef_root=None, ef_chunk_root=None, 
                 ef_parser='json', hathimeta=None, vecfile=None):
        '''
        ef_root: The root directory for Extracted Features files. Within that 
            directory, a pairtree file structure is observed.
        ef_chunk_root: The pairtree root directory for chunked Extracted Features
            files. This can be retrieved from a regular unchunked volume, but if you've
            preprocessed a chunked parquet-saved version of EF, specifying this is faster.
        ef_parser: The parser of the EF files. Likely 'json' or 'parquet'. Parquet is
            much quicker, if you've preprocessed the files. compare-tools/scripts/
            convert-to-parquet.py shows an example for that conversion.
        hathimeta: An initialized HathiMeta object. This is a DB-back metadata lookup.
        vecfile: An initialized Vectorfile (from PySRP), which contains vector representations
            of volumes. If you have multiple Vectorfile listings, this can be a list of 
            (key, Vectorfile) tuples. If the vectorfiles are by page or by chunk, their keys should
            be in the MTID notation.
        '''
        self.htid = htid
        
        self.ef_root = ef_root
        self.ef_parser = ef_parser
        self.ef_chunk_root = ef_chunk_root
        self._metadb = hathimeta
        self._meta = pd.Series()
        
        if vecfile:
            raise Exception("Not yet implemented!")
        
        self.reader = None
        self._volume = None
        self._tokensets = None
      
    def _ef_loc(self, scope='page'):
        loc = id_to_rsync(self.htid)
        if scope == 'chunk' and self.ef_chunk_root:
            loc = loc.replace('.json.bz2', '')
            return os.path.join(self.ef_chunk_root, loc)
        elif self.ef_root:
            if self.ef_parser == 'parquet':
                loc = loc.replace('.json.bz2', '')
            return os.path.join(self.ef_root, loc)
        else:
            raise Exception("Not enough EF information - set ef_root or ef_chunk_root.")

    @property
    def volume(self):
        if not self._volume:
            self._volume = Volume(self._ef_loc(scope='page'), parser=self.ef_parser)
        return self._volume
    
    @property
    def chunked_volume(self):
        if not self._chunked_volume:
            self._chunked_volume = Volume(self._ef_loc(scope='chunk'), parser='parquet')
        return self._chunked_volume

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
        if self._volume:
            # Note that this doesn't call self.volume: it will only show
            # the pretty HTML if a volume has already been initialized.
            return self.volume._repr_html_()
        else:
            try:
                return "<strong>%s</strong>" % (self.meta['title'])
            except:
                return self.htid
        
    def meta(self, reload=False):
        ''' Combine metadata from Hathifiles and Volumes, if available. '''
        if (self._meta.empty) or reload:
            if self._metadb:
                self._meta = self._metadb[self.htid].sort_index()

            if (self.ef_root or self.ef_chunk_root):
                meta2 = pd.Series(self.volume.parser.meta).sort_index()
                if not self._meta.empty:
                    self._meta = pd.concat([self._meta, meta2]).sort_index()
                else:
                    self._meta = meta2

        return self._meta
        
        
    def __str__(self):
        try:
            return "<HTRC Volume: %s - %s (%s)>" % (self.htid, self.meta['title'], self.meta['year'])
        except:
            return "<HTID: %s>" % self.htid
