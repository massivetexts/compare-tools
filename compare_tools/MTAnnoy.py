from SRP import Vector_file
from annoy import AnnoyIndex
import pandas as pd
import numpy as np

def create_annoy_index(filename, vector_filepaths, dims=300, n_trees=10, 
                       check_dupes=False):
    ''' 
    Build an Annoy index for approximate nearest neighbours, ingesting one
    or more of the pySRP vector files. Uses the on-disk build.
    
    Includes an index of ids numbers and mtids, saves as {filename}.index.pq
    '''
    import time
    start = time.time()

    if type(vector_filepaths) is not list:
        vector_filepaths = [vector_filepaths]
        
    t = AnnoyIndex(dims)
    t.on_disk_build(filename)
    
    # List of mtids, where the list index matches the index given to annoy
    ind = []
    unique = set()
    lasthtid = None
    
    i = 0
    for path in vector_filepaths:
        with Vector_file(path, mode='r') as vecf:
            assert dims == vecf.dims
            for ix, vec in vecf:
                if check_dupes:
                    # Does two things - avoids publicated pages / chunks,
                    # and only allows consecutive streams of a book - once
                    # the stream has moved on, that book can't be added again
                    htid, seq = ix.split('-')
                    if lasthtid != htid:
                        if htid in unique:
                            continue
                        else:
                            lasthtid = htid
                            unique.add(htid)
                            currentseqs = set([seq])
                    elif seq in currentseqs:
                        continue
                    else:
                        currentseqs.add(seq)
                        
                if blacklist and (ix.split('-')[0] in blacklist):
                    continue
                
                assert i == len(ind)
                i += 1
                ind.append(ix)
                t.add_item(i, vec)

        print("Total vecs", len(ind), end=',')

    print("Done ingesting. Time: %.0f seconds; Building" % (time.time() - start))
    t.build(n_trees)
    
    print("Done build. Time: %.0f seconds; Saving Index" % (time.time() - start))
    #ind = pd.Series(ind).to_frame('mtid')
    ind = (pd.Series(ind).apply(lambda x: x.split('-')[0])
                        .reset_index()
                        .rename(columns={0:'htid'})
                        .groupby('htid')['index'].aggregate(['min', 'max'])
                        .sort_index()
                       )
    ind.to_parquet('%s.index.pq' % filename, compression='snappy')
    
class MTAnnoy():
    '''
    Wrapper for Annoy that maintains HathiTrust HTID names.
    '''
    
    def __init__(self, annoypath, dims):
        self.u = AnnoyIndex(dims)
        self.u.load(annoypath)
        
        # This index expects books are in consecutive runs, since it only
        # only stores min annoy id and max annoy id
        self.ind = pd.read_parquet(annoypath+'.index.pq')
        
    def _find_htid(self, i):
        result = self.ind[(self.ind['min'] <= i) & (self.ind['max'] >= i)]
        assert result.shape[0] == 1
        return result.iloc[0]

    def get_htid_by_id(self, i):
        row = self._find_htid(i)
        return row.name

    def get_mtid_by_id(self, i):
        row = self._find_htid(i)
        htid = row.name
        seq = 1 + i - row['min']
        return "%s-%04.f" % (htid, seq)
    
    def get_id_by_htid(self, htid):
        return self.ind.loc[htid]

    def get_id_by_mtid(self, mtid):
        htid, seq = mtid.split('-')
        seq = int(seq)
        htidref = self.get_id_by_htid(htid)
        return htidref['min'] + seq - 1
    
    def get_nns_by_mtid(self, mtid, n, **kwargs):
        annoyid = self.get_id_by_mtid(mtid)
        return self.get_nns_by_item(annoyid, n, **kwargs)

    def get_nns_by_item(self, i, n, **kwargs):
        '''
        Wrapper around Annoy's get_nns_by_item which returns the mtids for the ids.
        The original method is under MTAnnoy().u.get_nns_by_item
        '''
        results = self.u.get_nns_by_item(i, n, **kwargs)
        if kwargs['include_distances']:
            results, distances = results
            
        named_results = [self.get_mtid_by_id(i) for i in results]
        
        if kwargs['include_distances']:
            return (named_results, distances)
        else:
            return named_results
        
    def get_named_result_df(self, mtid, n=30):
        ''' Return matches with distances and ranks, in tabular format'''
        j = 0
        rows = []
        r,d = self.get_nns_by_mtid(mtid, n, include_distances=True)

        for mtid,dist in zip(r,d):
            htid, seq = mtid.split('-')
            seq = int(seq)
            if not j:
                target = htid
                targetseq = seq
            else:
                rows.append((target, targetseq, htid, seq, dist, j))
            j += 1
        df = pd.DataFrame(rows, columns=['target', 'target_seq', 'match', 'match_seq', 'dist', 'match_rank'])
        return df