import os
from htrc_features import FeatureReader, Volume, resolvers
from htrc_features.feature_reader import group_tokenlist
from htrc_features.utils import id_to_rsync, _id_encode
import pandas as pd
import numpy as np
import json
import urllib
import warnings
from pathlib import Path

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


class StubbytreeResolver(resolvers.IdResolver):
    '''
    An alternative to pairtree that uses loc/code, where the code is every third digit of the ID.
    '''
    def __init__(self, dir=None, **kwargs):
        if not dir:
            raise NameError("You must specify a directory with 'dir'")
        super().__init__(dir=dir, **kwargs)
            
    def id_to_stubbydir(self, htid):
        '''
        Take an HTRC id and convert it to a 'stubbytree' location.

        '''
        libid, volid = htid.split('.', 1)
        volid_clean = _id_encode(volid)

        path = os.path.join(libid, volid_clean[::3])

        return path
        
    def _open(self, id, mode = 'rb', format=None, compression='default', suffix=None, **kwargs):
        assert(mode.endswith('b'))
        
        if not format:
            format = self.format
        if compression == 'default':
            compression = self.compression

        path = self.id_to_stubbydir(id, format, suffix, compression)
        fname = self.fname(id, format= format, suffix = suffix, compression = compression)
        full_path = Path(self.dir, path, fname)
        try:
            return full_path.open(mode=mode)
        except FileNotFoundError:
            if mode.startswith('w'):
                full_path.parent.mkdir(parents=True, exist_ok=True)
                return full_path.open(mode=mode)
            else:
                raise


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
    def __init__(self, htid, id_resolver, chunked_resolver = None, hathimeta=None, vecfiles=None):
        '''
        id_resolver: An IdResolver for Extracted Features files.
        chunked_resolver: An IdResolver chunked Extracted Features
            files. This can be retrieved from a regular unchunked volume, but if you've
            preprocessed a chunked parquet-saved version of EF, specifying this is faster.
        hathimeta: An initialized HathiMeta object. This is a DB-back metadata lookup.
        vecfiles: An initialized Vectorfile (from PySRP), which contains vector representations
            of volumes. You can provide just a single vectorfile, a tuple of
            (key, Vectorfile), or a list with multiple (key, Vectorfile) tuples.
            Currently, only vectorfiles by mtid are supported - the htid and a 
            four character wide one-indexed integer.
        '''
        self.htid = htid
        self.id_resolver = id_resolver
        if chunked_resolver is None:
            self.chunked_resolver = id_resolver
        else:
            self.chunked_resolver = chunked_resolver
        self._metadb = hathimeta
        self._meta = pd.Series()
        
        if not vecfiles:
            self._vecfiles = []
        elif type(vecfiles) is tuple:
            self._vecfiles = [vecfile]
        elif type(vecfiles) is not list:
            self._vecfiles = [('vectors'), vecfiles]
        elif (type(vecfiles) is list) and (type(vecfiles[0]) is tuple):
            self._vecfiles = vecfiles
        else:
            raise Exception("Unexpected vecfile input format")
            
        self.vecnames = [n for n, vfile in self._vecfiles]
        
        self.reader = None
        self._volume = None
        self._chunked_volume = None
        self._tokensets = dict(page=pd.DataFrame(), chunk=pd.DataFrame())
        self._vecfile_cache = dict()

    def vectors(self, keyfilter=None, drop_nan=True):
        ''' Return a list of key, mtids, vectors for each available vectorfile.
            If you are interested in just one of the vectorfiles, set keyfilter
            to it's name. Data is internally cached.
            
            drop_nan: Drop rows that have a NaN anywere. These shouldn't be in the
                vector files, so you can turn off the dropping in order to find 
                upstream problems.
        '''
        allvecs = []
        for name, vecfile in self._vecfiles:
            if keyfilter and (name != keyfilter):
                continue
            if name not in self._vecfile_cache:
                mtids, vectors = self._get_mtid_vecs(vecfile)
                self._vecfile_cache[name] = (mtids, vectors)
                
            mtids, vectors = self._vecfile_cache[name]
            
            if drop_nan:
                no_nan = ~np.isnan(vectors).any(axis=1)
                mtids = mtids[no_nan]
                vectors = vectors[no_nan]
            
            if keyfilter:
                return mtids, vectors
            else:
                allvecs.append((name, mtids, vectors))

        if keyfilter:
            raise KeyError("No vecfile with that name. If you didn't specific a name, the default name is 'vectors'")
        return allvecs

    @property
    def volume(self):
        if not self._volume:
            self._volume = Volume(self.htid, id_resolver = self.id_resolver)
        return self._volume
    
    @property
    def chunked_volume(self):
        if not self._chunked_volume:
            self._chunked_volume = Volume(self.htid, id_resolver = self.chunked_resolver)
        return self._chunked_volume
    
    def _get_mtid_vecs(self, vecfile):
        ''' Retrieve mtid formatted vectors (using old htid-0001 format) from the 
        given vectorfile.'''
        i = 1
        vecs = []
        mtids = []
        while True:
            try:
                mtid = '{}-{:04d}'.format(self.htid, i)
                vecs.append(vecfile[mtid])
                mtids.append(mtid)
            except KeyError:
                break
            i += 1

        if i == 1:
            raise Exception("Got to work with the new format using the new code... hold on.")

        return np.array(mtids), np.vstack(vecs)

    def tokenlist(self, scope='chunk', **kwargs):
        """
        Return a pandas frame with page, token, and count information.
        """
        if scope == 'page':
            return self.volume.tokenlist(case=False, pos=False, **kwargs)
        elif scope == 'chunk':
            return self.chunked_volume.tokenlist(case=False, pos=False, chunk = True, 
                                                         suppress_warning=True, **kwargs)
    
    def chunked_tokenlist(self, **kwargs):
        """
        Return a pandas frame with page, token, and count information.
        """
        return self.volume.chunked_tokenlist(case=False, pos=False, **kwargs)
    
    def tokensets(self, scope='chunk'):
        ''' Sets of tokens per chunk/page. Scope can be pages or chunks. Chunks by default. '''
        if self._tokensets[scope].empty:
            if scope == 'chunk':
                self._tokensets['chunk'] = (self.tokenlist('chunk').reset_index()
                                            .groupby('chunk')
                                            .apply(lambda x: set(x['lowercase']))
                                           )
            elif scope == 'page':
                self._tokensets['page'] = (self.tokenlist('page').reset_index()
                                           .groupby(self.volume._pagecolname)
                                           .apply(lambda x: set(x['lowercase']))
                                          )
            else:
                raise Exception("Unknown scope")

        return self._tokensets[scope]
    
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
        
    def meta(self, reload=False, dedupe=False):
        ''' Combine metadata from Hathifiles and Volumes, if available. Keys are not checked for
        duplication - if that occurs, (e.g. two 'page_count' rows), the first is from the
        Hathifiles and the second (if dedupe=False) is from the volume.
        '''
        if (self._meta.empty) or reload:
            if self._metadb:
                self._meta = self._metadb[self.htid].sort_index()

            try:
                meta2 = pd.Series(self.volume.parser.meta).sort_index()
                if not self._meta.empty:
                    self._meta = pd.concat([self._meta, meta2]).sort_index()
                else:
                    self._meta = meta2
            except:
                pass
        if not "page_count" in self._meta:
            self._meta["page_count"] = 1
                
        if not dedupe:
            return self._meta
        else:
            return self._meta.loc[~self._meta.index.duplicated(keep='first')]
        
        
    def __str__(self):
        try:
            return "<HTRC Volume: %s - %s (%s)>" % (self.htid, self.meta['title'], self.meta['year'])
        except:
            return "<HTID: %s>" % self.htid
