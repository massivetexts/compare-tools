from SRP import Vector_file
from annoy import AnnoyIndex
import pandas as pd
import numpy as np
from compare_tools.utils import split_mtid
import os

def create_annoy_index(filename, vector_filepaths, dims=300, n_trees=10, 
                       check_dupes=False, on_disk=True):
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
    if on_disk:
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
                norm = np.linalg.norm(vec)
                if norm==0 or np.isnan(norm) or np.isinf(norm):
                    continue
                vec = vec/norm
                if check_dupes:
                    # Does two things - avoids duplicated pages / chunks,
                    # and only allows consecutive streams of a book - once
                    # the stream has moved on, that book can't be added again
                    mtid_split = split_mtid(ix)
                    htid = mtid_split[0]
                    seq = "-".join([str(x) for x in mtid_split[1:]])

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
                
                assert i == len(ind)
                ind.append(ix)
                t.add_item(i, vec)
                i += 1

        print("Total vecs", len(ind), end=',')

    print("Done ingesting. Time: %.0f seconds; Building" % (time.time() - start))
    t.build(n_trees)
    
    if not on_disk:
        t.save(filename)
    
    print("Done build. Time: %.0f seconds; Saving Index" % (time.time() - start))
    #ind = pd.Series(ind).to_frame('mtid')
    
    ind = (pd.Series(ind).apply(lambda x: x.split('-', 1)[0])
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
    
    def __init__(self, annoypath, dims, prefault=False):
        self.u = AnnoyIndex(dims, metric='angular')
        self.u.load(annoypath, prefault=prefault)
        
        # This index expects books are in consecutive runs, since it only
        # only stores min annoy id and max annoy id
        self.ind = pd.read_parquet(annoypath + '.index.pq')
        self.ind['length'] = self.ind['max'] - self.ind['min'] + 1
        
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
        htid, seq = split_mtid(mtid)
        htidref = self.get_id_by_htid(htid)
        return htidref['min'] + seq - 1
    
    def get_nns_by_mtid(self, mtid, n, **kwargs):
        annoyid = self.get_id_by_mtid(mtid)
        return self.get_nns_by_item(annoyid, n, **kwargs)

    def get_nns_by_item(self, i, n, include_distances=False, search_k=-1, **kwargs):
        '''
        Wrapper around Annoy's get_nns_by_item which returns the mtids for the ids.
        The original method is under MTAnnoy().u.get_nns_by_item
        '''
        results = self.u.get_nns_by_item(i, n, include_distances=include_distances, search_k=search_k, **kwargs)
        if include_distances:
            results, distances = results
            
        named_results = [self.get_mtid_by_id(i) for i in results]
        
        if include_distances:
            return (named_results, distances)
        else:
            return named_results
        
    def _result_df(self, i=None, n=30, max_dist=None, rank=True, search_k=-1):
        ''' Quick id-only lookup, returning a target_i/match_i/dist DataFrame'''
        r, d = self.u.get_nns_by_item(i, n, include_distances=True, search_k=search_k)
        df = pd.DataFrame([(i, match_i, dist) for match_i, dist in zip(r,d)],
                          columns=['target_i', 'match_i', 'dist'])
        if max_dist:
            df = df[df.dist <= max_dist]
        if rank:
            df = df.reset_index().rename(columns={'index':'rank'})
            df = df[['target_i', 'match_i', 'rank', 'dist']]
        return df
        
    def _result_df_by_htid(self, htid, n=30, rank=True, max_dist=None, dedupe=True, search_k=-1):
        details = self.ind.loc[htid]
        vol_df = []

        for i in range(details['min'], details['max'] + 1):
            df = self._result_df(i, n=n, max_dist=max_dist, rank=rank, search_k=search_k)
            vol_df.append(df)
        df = pd.concat(vol_df)

        if dedupe:
            # Dedupe so that each matching chunk can only match with one 
            # chunk from the volume (i.e. one-to-one)
            df = (df.groupby('match_i', as_index=False)
                            .apply(lambda x: x.sort_values('dist').iloc[:1])
                            .sort_values(['target_i', 'rank'])
                            .droplevel(1)
                  )
        return df

    def get_named_result_df(self, i=None, mtid=None, htid=None, n=30, dedupe=False, max_dist=None, search_k=-1):
        ''' Return matches with distances and ranks, in tabular format.'''

        try:
            # only one can be set
            assert len([1 for val in [i,mtid,htid] if val])
        except:
            raise AssertionError('Need one (and only one) of i, mtid, or htid')
        if mtid:
            i = self.get_id_by_mtid(mtid)
        if i: # Gets a single item.
            df = self._result_df(i, n=n, max_dist=max_dist, rank=True, search_k=search_k)
        elif htid:
            df = self._result_df_by_htid(htid, n=n, rank=True, max_dist=max_dist, dedupe=dedupe, search_k=search_k)
        
        # Expand item ids to hathitrust id + seq. Replace() is for performance, though
        # really this process would be faster if something like 'keys' ( {'htid': id} for 
        # all vectors) was internally kept. I'm not confident that it would scale to 7m,
        # though - the memory use might preclude multi-processing
        unique = np.concatenate([df.target_i.unique(), df.match_i.unique()])
        keys = { i: self.get_mtid_by_id(i) for i in unique }
        df[['target', 'target_seq']] = df.target_i.replace(keys).str.split('-', expand=True)
        df[['match', 'match_seq']] = df.match_i.replace(keys).str.split('-', expand=True)
        df[["match_seq", "target_seq"]] = df[["match_seq", "target_seq"]].apply(pd.to_numeric)
        
        return df[['target', 'target_seq', 'match', 'match_seq', 'dist', 'rank']]
    
    def doc_match_stats(self, htid, n=30, min_count=None, max_dist=None, search_k=-1):
        '''
        Return a summed DF, that reduces chunk to chunk matches to doc to doc matching statistics:
            - count of matching chunks
            - mean similarity for matched chunks
            - proportion of the target that the candidate matches, and vice-versa
        
        min_count - minimum number of matches between left and right to keep. In the edge 
            case where the left has less than that (e.g. if left only has one chunk and you have a min_count of
            2), the min_count is set to the number of left chunks for book.
        '''
        target_length = self.ind.loc[htid].length
        if min_count and min_count > target_length:
            min_count = target_length
        if min_count and min_count == 1:
            min_count = None

        df = self.get_named_result_df(htid=htid, n=n, max_dist=max_dist, dedupe=True, search_k=search_k)
        
        stats = df.groupby(['target', 'match'])['dist'].aggregate(['count', 'mean']).sort_values(['count', 'mean'], ascending=False).reset_index(0)
        stats = pd.merge(stats, self.ind.loc[stats.index].length, right_index=True, left_index=True)
        stats['prop_target'] = stats['count'] / target_length
        stats['prop_match'] = stats['count'] / stats.length
        stats = stats.reset_index()
        if min_count:
            stats = stats[stats['count'] >= min_count]
        return stats

    
if __name__ == '__main__':
    pass


class TitleAnnoy():

    def __init__(self, path, dims=50):
        '''
        AnnoyIndex for quick fuzzy matching of titles. Uses a subword byte-pair-encoding
            vector space model called BPEmb.
        '''
        self.index_path = path
        self.index_reference_path = os.path.splitext(path)[0] + '-ref.csv.gz'
        self.u = AnnoyIndex(dims, 'angular')
        self.dims = dims
        
    def load(self, prefault=False):
        self.u.load(self.index_path, prefault=prefault)
        
        df_ref = pd.read_csv(self.index_reference_path, index_col=0, compression='gzip')
        self.id2htid = df_ref['htid'].to_dict()
        self.htid2id = df_ref['htid'].reset_index().set_index('htid')['index'].to_dict()
        
    def build_index(self, meta_parquet_path, vocab_size=50000, trees=100):
        '''
        Build an annoy index of metadata titles. Uses BPEEmb, so small vocab size is fine
        
        index_path: path with filename, where filename ends with '.ann'
        trees: Number of trees to use for the Annoy index. More is better but slower to
            build.
        '''
        from bpemb import BPEmb
        from compare_tools.hathimeta import clean_title
        
        metadf = pd.read_parquet(meta_parquet_path, columns=['htid', 'title'])
        bpemb_en = BPEmb(lang="en", dim=self.dims, vs=vocab_size)

        # Insert vectors for documents into Annoy index, using the integer from
        # the metadf index as the id
        for i, row in metadf.reset_index().fillna('').astype(str).iterrows():
            bpe_ids = bpemb_en.encode_ids(row.title)
            # Sum of full title. Imperfect, would work better if BPEs for each word were averaged first.
            vec = bpemb_en.vectors[bpe_ids].sum(0)
            
            trimmed_bpe_ids = bpemb_en.encode_ids(clean_title(row.title))
            trimmed_vec = bpemb_en.vectors[trimmed_bpe_ids].sum(0)
            
            # Average, with more weight on the cleaned title.
            weighted = np.average([vec, trimmed_vec], axis=0, weights=[.3, .7])
            
            self.u.add_item(i, weighted)
            if i % 100000 == 0:
                print(i, end=',')
        print()

        # will take about 30m for 100 dims and 8mi titles
        self.u.build(trees)
        self.u.save(index_path)
        metadf.reset_index()['htid'].to_csv(self.index_reference_path, compression='gzip')