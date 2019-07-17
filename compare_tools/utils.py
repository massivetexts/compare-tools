import os
import pandas as pd
from htrc_features import FeatureReader
from htrc_features.feature_reader import group_tokenlist
from htrc_features.utils import id_to_rsync
import pandas as pd
import numpy as np
import json
import urllib
import altair as alt
alt.data_transformers.enable('json')

def split_mtid(mtid):
    htid, seq = mtid.split('-')
    seq = int(seq)
    return htid, seq
    
def join_mtid(htid, seq):
    return "$s-%04.f" % (htid, seq)

def page_counts(id, hathi_loc = "../../hathi-ef/"):
    loc = hathi_loc + id_to_rsync(id)
    parquet_loc = loc.replace(".json.bz2", ".parquet")
    if os.path.exists(parquet_loc):
        return pd.read_parquet(parquet_loc)
    vol = list(FeatureReader([loc]).volumes())[0]
    table = vol.tokenlist().groupby(["page", "token"])['count'].sum().reset_index()
    table.to_parquet(parquet_loc)
    return table

class Comparison(object):

    """
    A comparison has two elements; left and right.
    Different classes will be initialized from different sorts of sources.
    """
    
    def __init__(self, left, right, labels = ['left', 'right']):
        """
        initialized with two dataframes columned ['page/chunk/etc', 'token', 'count']
        """
        self.left = left
        self.right = right

    def build_smith_waterman_matrix(self, threshold):
        """
        Build a cumulative Smith-Waterman distance matrix to find contiguous runs.

        threshold: any similarities below this will be counted 
        as breaks in the alignment.

        Uses the matrix at 'self.similarity_matrix', which may need to be tweaked.
        Note that this is a **similarity** matrix between the two sets. Must be
        a similarity matrix, *not* a distance matrix!

        """

        # This matrix will track the 
        # size of the longest continguous run leading to any point
        dm = self.similarity_matrix
        sw = np.zeros_like(dm)
        
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                # This will occasionally wrap around from the first
                # row or column to the last one.
                # but the values there are guaranteed to be 
                # zero, so that's OK. 

                # It also includes the current cell to guarantee a zero,
                # so that no cell is penalized for having low neighbors.

                max_neighbor = np.max(sw[(i, i, i-1, i-1), (j, j-1, j, j-1)])
                if pd.isna(dm[i, j]):
                    continue
                if dm[i, j] > threshold:
                    sw[i, j] = dm[i, j] - threshold + max_neighbor
                # The condition above breaks chains immediately.
                
        self.sw_scores = sw
        return self.sw_scores
    
    def yield_sw_runs(self, threshold, no_joined_chunks = True):
        self.sw_copy = np.copy(self.sw_scores)
        while True:
            next_up = self.return_top_sw_sequence_and_remove_from_matrix(threshold)
            if next_up is None:
                # We've exhausted the supply of matches.
                break
            if next_up[-1][2] == np.Infinity:
                if no_joined_chunks:
                    # Suppress runs that are contiguous to previous runs.
                    continue
                else:
                    # Strip off that thing that's contiguous to the old chain.
                    yield next_up[:-1]
            yield next_up

    def assemble_sw_runs(self, mat = "similarity_matrix", threshold = 0.05, min_length = 3, min_strength = 0):
        self.build_smith_waterman_matrix(threshold)
        self.sw_copy = np.copy(self.sw_scores)
        self.similarity_copy = np.copy(self.similarity_matrix)

        runs = []
        run_num = 0
        for run in self.yield_sw_runs(threshold):
            if run is None:
                # The iteration was stopped.
                break
            length = len(run)
            strength = sum([item[2] for item in run])
            if length > min_length and strength > min_strength:
                # Add the run num.
                for (l, r, sim) in run:
                    runs.append((l, r, sim, run_num))
            run_num += 1
        self.runs = pd.DataFrame(runs, columns=['left_seq', 'right_seq', 'sim', 'run_num'])

    def export_runs(self, output, **kwargs):
        """

        Only works with one-htid to one-htid vector comparisons.
        """
        if not hasattr(self, "runs"):
            self.assemble_sw_runs(**kwargs)
            
        self.runs['left_htid'] = self.left_htid
        self.runs['right_htid'] = self.right_htid
        self.runs.to_csv(output, mode="a", header = False, index = False)
        
    def return_top_sw_sequence_and_remove_from_matrix(self, threshold):
        """
        This code does two things.

        1. It extracts the longest sequence in the Smith Waterman matrix.
        2. It fixes the matrix so that later searches won't return stretches addressing
           the same span of text.

        threshold: the minimum similarity score across which to follow a run.
                   This will be very different depending on the metric used.
                   For raw SRP cosine, maybe 0.7. For document-adjusted, maybe 0.05.
        """
        if not 'sw_copy' in self.__dict__:
            self.sw_copy = np.copy(self.sw_scores)
            
        scores = self.sw_copy
        mat = self.similarity_matrix
        
        # Start at the end of the longest chain.

        # Mask out the infinite placeholders.
        finite_mat = np.copy(scores)
        
        finite_mat[np.isinf(scores)] = 0

        maxed = finite_mat.argmax()

        x, y = np.array(np.unravel_index(maxed, scores.shape))
        
        if np.isinf(self.sw_copy[(x, y)]):
            return None
        
        # Initialize the list with that point.
        l = [(x, y, mat[(x, y)])]

        # And zero it out for next time.
        
        while mat[(x, y)] > 0:

            # Examine three cells; above, left, and above+left.
            
            choices = np.array([(x-1, y), (x, y-1), (x-1, y-1)])

            # The possibilities are the distances at each of the points.
            vals = self.similarity_copy[tuple(choices.T)]
            matched = False
            # Move over the choices in order from largest to smallest.
            for ix in np.argsort(-vals):
                x_, y_ = choices[ix]
                val = vals[ix]
                
                # This > 0 check is necessary because numpy might be wrapping
                # around the edge of a matrix. If it is, we just ignore the entry.
                
                if x_ >= 0 and y_ >= 0 and val > threshold:
                    x = x_
                    y = y_
                    l.append((x, y, val))
                    if scores[(x, y)] == np.Infinity:
                        l[-1] = (x, y, np.Infinity)
                        return l
                    
                    # Alert that we're still going.
                    matched = True
                    # break now b/c we don't need to consider smaller valaues.
                    break
            if matched:
                # Continue looking at the matrix. 
                continue
            # If we didn't find a match, we're done.
            break
        
        self.sw_copy = scores
        # Zero out 
        for i, (x, y, score) in enumerate(l):
            # Zero out both cumulative and absolute scores.
            scores[x, :] = 0
            self.similarity_copy[x, :] = 0
            scores[:, y] = 0
            self.similarity_copy[:, y] = 0
            
            scores[(x, y)] = np.Infinity
            
        return l        

    @property
    def jaccard_matrix(self, document = "seq"):
        """
        Return pairwise jaccard similarities across pages.

        'document': The column in the initialized dataframe containing page-level info. 
        """

        if hasattr(self, "_jaccard_mat"):
            return self._jaccard_mat
        output = []
        mat = np.full((len(self.left.tokensets), len(self.right.tokensets)), np.nan)
        for i, (left_id, left_set) in enumerate(self.left.tokensets.iteritems()):
            # Cache the left length
            left_length = len(left_set)
            for j, (right_id, right_set) in enumerate(self.right.tokensets.iteritems()):
                if i <= j:
                    inter = len(left_set.intersection(right_set))
                    sim = inter/(len(right_set) + left_length - inter)
                    mat[i, j] = sim
                    mat[j, i] = sim
                    
        self._jaccard_mat = mat
        self.similarity_matrix = mat
        return mat

    def plot(self, which = None, ids = None, runs = None, filter = None, width = 500, height = 500):
        if which is None:
            which = "similarity_matrix"
        
        mat = getattr(self, which)

        vals = []

        ix = 0
        for a in range(mat.shape[0]):
            for b in range(mat.shape[1]):
                if filter:
                    if not filter(mat[a,b]):
                        continue
                if pd.isna(mat[a, b]):
                    continue
                vals.append((a, b, mat[a,b], ix))
                ix += 1
        data = pd.DataFrame(vals, columns=['left_seq', 'right_seq', 'sim', 'ix'])

        if ids:
            labs = [(ida, idb) for ida in ids[0] for idb in ids[1]]

        colorscale = alt.Color('sim')

        if which == 'similarity_matrix':
            colorscale = alt.Color('sim', scale=alt.Scale(zero=False, domain = [.2, 1]))
    
        if which == 'similarity_matrix' and self.adjusted:
            colorscale = alt.Color('sim', scale=alt.Scale(zero=False, domain = [-.05, .3]))

        grid = alt.Chart(data).mark_rect().encode(
            x = "left_seq:O",
            y="right_seq:O",
            color=colorscale,
            tooltip='sim'
        )

        charts = [grid]
        if runs is not None:
            run_chart = alt.Chart(self.runs).encode(
                x = "left_seq:O",
                y = "right_seq:O",
                detail = 'run_num',
                tooltip = 'sim'
            ).mark_line(color='red')
            return grid + run_chart
            charts.append(run_chart)
            
        return alt.layer(*charts).properties(height=600, width=600)


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
    def __init__(self, htid, rsync_root = "../../hathi-ef"):
        self.htid = htid
        self.reader = None
        self._volume = None
        self._tokensets = None
        self.rsync_root = rsync_root
        
    def _rsync_loc(self):
        loc = id_to_rsync(self.htid)
        return self.rsync_root + loc

    @property
    def volume(self):
        if self._volume is not None:
            return self._volume
        if self.reader is None:
            self.reader = FeatureReader(self._rsync_loc())
        self._volume = self.reader.first()
        return self._volume

    @property
    def page_counts(self):
        # This could wrap around parquet.
        return self.volume.tokenlist().groupby(["page", "token"])['count'].sum().reset_index()

    @property
    def tokensets(self):
        if self._tokensets is not None:
            return self._tokensets
        self._tokensets = self.page_counts.groupby("page").apply(lambda x: set(x['token']))
        return self._tokensets
    
    def _repr_html_(self):
        return self.volume._repr_html_()

class EFComparison(Comparison):
    def __init__(self, left, right, rsync_root = "../../hathi-ef"):
        """
        Initialize with either an HTID object;
        or with two HTID strings.

        No mixing and matching; both must be the same type.

        "rsync_root" is passed to HTID.
        """
        if hasattr(left, "page_counts"):
            print("Using HTID class")
            self.left = left
            self.right = right
        else:
            print("Initializing from string")
            self.left = HTID(left, rsync_root)
            self.right = HTID(right, rsync_root)
        
    
class VectorComparison(Comparison):
    """
    A comparison between two vectorized representations of texts (GloVe, SRP, etc.)
    

    """
    def __init__(self, left, right, corpus, adjusted = True):
        """
        left: a list of mtids or htids
        right: a list of mtids or htids
        corpus: a ChunkCollection object.
        """
        if isinstance(left, str):
            self.left_htid = left
            left = [left]
        if isinstance(right, str):
            self.right_htid = right            
            right = [right]            
            
        self.adjusted = adjusted
        
        mat, self.ids = corpus.paired_distance(
            left, right,
            format='matrix', adjusted = adjusted)

        self.similarity_matrix = mat