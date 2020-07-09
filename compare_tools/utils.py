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
import uuid
from SRP import Vector_file
import SRP
from .configuration import config

# These are global lists to allow thread-safe creation of new vectorfiles..
# They'll be filled inside processes.
vector_files = {
    'SRP':[],
    'glove': []
}

try:
    from .configuration import rsync_root
except:
    warnings.warn("No rsync root found")
    # I don't think I wrote this; but I'm not sure this necessarily follows? --BMS
    parquet_root = None

def split_mtid(mtid):
    parts = mtid.split('-')
    diff = len(parts) - 4
    if diff > 0:
        # Adjust for when htid has an underscore - though upstream and downstream tasks may not support this
        parts = ["-".join(parts[:1+diff]), parts[1+diff:]]
    htid, seq = parts[:2]

    if len(parts) > 2:
        start, end = parts[2:]
        start = int(start)
        end = int(end)
        return htid, seq, start, end
    else:
        return htid, seq

def htid_ize(mtid):
    """
    Extract the MTID from an HTID.
    """
    return mtid.split("-")[0]

hasher = None
def SRP_transform(f):
    global hasher
    if hasher is None:
        hasher = SRP.SRP(640)
    return hasher.stable_transform(words = f['token'], counts = f['count'], log = True, standardize = True)

def supplement_vectors(volume, *vector_files):
    from .wem_hook import WEM_transform
    
    print("Supplementing")
    # Takes a volume and any number of vector files.
    
    chunks = volume.tokenlist(chunk = True, chunk_size = 10000, overflow = 'ends', pos=False, page_ref = True)
    chunks.reset_index(level = 3, inplace = True)
    if chunks.empty:
        return
    my_chunks = []
    for (chunk, start, end) in set(chunks.index):
        my_chunks.append((volume.id, chunk, start, end, chunks.loc[(chunk, start, end)].reset_index(drop = True)))
    for file in vector_files:
        assert file.mode in ["a", "w"]
        if file.dims == 300:
            vtype = "glove"
        elif file.dims == 640:
            vtype = "SRP"
        else:
            raise NotImplementedError(f"Not sure how to handle a vector file like {file}")
        for (htid, chunk, start, end, group) in my_chunks:                                  
            mtid = "{}-{:04d}-{}-{}".format(htid, chunk, start, end)
            if vtype == "glove":
                file.add_row(mtid, WEM_transform(group).astype("<f4"))
            elif vtype=="SRP":
                file.add_row(mtid, SRP_transform(group))
        file.flush()


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
    def __init__(self, htid, id_resolver, chunk_resolver = None, hathimeta=None, vecfiles=None):
        '''
        id_resolver: An IdResolver for Extracted Features files.
        chunk_resolver: An IdResolver to chunked Extracted Features
            files. This can be retrieved from a regular unchunked volume, but if you've
            preprocessed a chunked parquet-saved version of EF, specifying this is faster.
        hathimeta: An initialized HathiMeta object. This is a DB-back metadata lookup.
        vecfiles: An initialized Vectorfile (from PySRP), which contains vector representations
            of volumes. You can provide just a single vectorfile, a tuple of
            (key, Vectorfile), or a list with multiple (key, Vectorfile) tuples.
            Currently, only vectorfiles by mtid are supported - the htid and a 
            four character wide one-indexed integer.
            
            If a vectorfile name has an underscore, it is considered a fallback - e.g. glove_backup
            is the fallback for glove. This can be used for including a set of vectorfiles.
            e.g. 'glove_1', 'glove_2', ..., 'glove_n'
        '''
        self.htid = htid
        self.id_resolver = id_resolver
        if chunk_resolver is None:
            self.chunked_resolver = id_resolver
        else:
            self.chunked_resolver = chunk_resolver
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
        for name, _ in self._vecfiles:
            if keyfilter and (name != keyfilter):
                continue
            if name not in self._vecfile_cache:
                mtids, vectors = self._get_mtid_vecs(name)
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
    
    def _get_mtid_vecs(self, name, write_missing=True, write_path='.'):
        ''' Retrieve mtid formatted vectors (using htid-0001-3-5 format) from the 
        given vectorfile.
        
        The 'name' refers to the name associated with the tuple provided when HTID was initialized.
        '''

        vecs = []
        mtids = []
            
        vecnames = sorted([vecname for vecname in self.vecnames if vecname.split('_')[0] == name])
        assert len(vecnames) > 0
        
        vals = []
        for name in vecnames:
            vecf = self._vecfiles[self.vecnames.index(name)][1]
            try:
                for mtid, vec in vecf.find_prefix(self.htid, "-"):
                    vals.append((mtid, vec))
            except:
                continue
            if len(vals) > 0:
                break
        try:
            vals.sort()
        except ValueError:
            print("There is a sorting error on the list of (mtid,vec) references. This is likely because "
                  "of duplicate mtid keys - when that happens, sorting tries to sort by vecs - but you shouldn't "
                 " have duplicates!")
            raise
        
        if len(vals) > 0:
            mtids, vecs = zip(*vals)
            return np.array(mtids), np.vstack(vecs)
        else:
            raise IndexError('{} not in {}'.format(self.htid, vecnames))

        '''
        # Temporarily cordoning off
        if (len(vals) == 0) and write_missing:
            try:
                # See if any files are open for writing
                writable = [f for f in vector_files if f.mode=='a'][0]
            except IndexError:
                suffix = uuid.uuid4().hex[:8]
                fname = Path(write_path, name + "-" + suffix + ".bin")
                writable = SRP.Vector_file(fname, mode = 'a', dims = vector_files[format][0].dims)
                vector_files[format].append(writable)

            supplement_vectors(self.volume, writable)

            # It just shouldn't be very expensive to do the lookup after doing the write
            # given how expensive the whole process is.
            mtids, vecs = zip(*writable.find_prefix(self.htid, "-"))
        '''
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

        
def concatenate_vector_files(inpath1, inpath2, outpath, newdim):
    '''
    Create a vector file that merges two other vectorfiles pairwise. It assumed that
    each row in input into each vector file in alignment. This is used for testing Glove+SRP together.
    '''
    i = 0
    vecf1 = Vector_file(inpath1, mode='r', offset_cache=False)
    vecf2 = Vector_file(inpath2, mode='r', offset_cache=False)
    with Vector_file(outpath, mode='w', dims=newdim) as newvecf:
        for result1, result2 in zip(vecf1, vecf2):
            mtid1, vec1 = result1
            mtid2, vec2 = result2
            if (mtid1 != mtid2):
                print("Not entirely aligned!", i, mtid1, mtid2)
            newvec = np.concatenate([vec1, vec2])
            newvecf.add_row(mtid2, newvec)
            i += 1
            if i % 1000 == 0:
                print(i, end=',')

    with Vector_file(outpath, mode='a', offset_cache=True) as newvecf:
        print('Building new offset lookup')
        newvecf._build_offset_lookup(sep='-', force=True)