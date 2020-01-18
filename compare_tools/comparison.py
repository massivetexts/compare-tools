import numpy as np
import pandas as pd
import altair as alt
alt.data_transformers.enable('json')

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

        Uses the matrix at 'self.similarity_matrix()', which may need to be tweaked.
        Note that this is a **similarity** matrix between the two sets. Must be
        a similarity matrix, *not* a distance matrix!

        """

        # This matrix will track the 
        # size of the longest continguous run leading to any point
        dm = self.similarity_matrix()
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
        self.similarity_copy = np.copy(self.similarity_matrix())

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
        mat = self.similarity_matrix()
        
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
        mat = np.full((len(self.left.tokensets()), len(self.right.tokensets())), np.nan)
        for i, (left_id, left_set) in enumerate(self.left.tokensets().iteritems()):
            # Cache the left length
            left_length = len(left_set)
            for j, (right_id, right_set) in enumerate(self.right.tokensets().iteritems()):
                if True: # i <= j:
                    inter = len(left_set.intersection(right_set))
                    sim = inter/(len(right_set) + left_length - inter)
                    mat[i, j] = sim
#                    mat[j, i] = sim
                    
        self._jaccard_mat = mat
        self.similarity_matrix = mat
        return mat

    def plot(self, which = None, ids = None, runs = None, filter = None, width = 500,
             height = 500, scale_domain= None):
        '''
        
        scale_domain: an optional 'domain' argument to Altair's color scale. Use 'unaggregated'
        to size to the data.
        '''
        
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
            domain = scale_domain if scale_domain else [-.02, 1.]
            colorscale = alt.Color('sim',
                                   scale=alt.Scale(zero=False, domain=domain))
    
        if which == 'similarity_matrix' and self.adjusted:
            domain = scale_domain if scale_domain else [-.05, .3]
            colorscale = alt.Color('sim', 
                                   scale=alt.Scale(zero=False, domain = domain))

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


class EFComparison(Comparison):
    def __init__(self, left, right, rsync_root=None, parquet_root=None):
        """
        Initialize with either an HTID object;
        or with two HTID strings.

        No mixing and matching; both must be the same type.

        "parquet_root" is passed to HTID.
        """
        if hasattr(left, "tokenlist"):
            print("Using HTID class")
            self.left = left
            self.right = right
        else:
            print("Initializing from string")
            self.left = HTID(left, rsync_root=rsync_root, parquet_root=parquet_root)
            self.right = HTID(right, rsync_root=rsync_root, parquet_root=parquet_root)
        
class HTIDComparison(Comparison):
    '''
    A comparison class that assumes both EF and Vector info is provided to HTID
    '''
    
    def __init__(self, left, right, labels = ['left', 'right']):
        self.left = left
        self.right = right
        self._distance_matrix = dict()
    
    def similarity_matrix(self, vecname='vectors', include_index=False, **kwargs):
        '''
        Vecname refers to the name of the Vector_file in HTID. If unnamed, HTID calls
        it 'vectors'. 'glove' and 'srp' are used for the SaDDL project.
        
        Inverse of distance matrix, so ***assumes distance metric is bounded 0-1***
        '''
        dist = self.distance_matrix(vecname, include_index, **kwargs)
        return 1 - dist
        
    def distance_matrix(self, vecname='vectors', include_index=False, **kwargs):
        '''
        Vecname refers to the name of the Vector_file in HTID. If unnamed, HTID calls
        it 'vectors'. 'glove' and 'srp' are used for the SaDDL project.
        '''
        from scipy.spatial.distance import cdist
        
        if (vecname not in self._distance_matrix) or not self._distance_matrix[vecname]:
            leftids, leftvecs = self.left.vectors('glove')
            rightids, rightvecs = self.right.vectors('glove')
            sims = cdist(leftvecs, rightvecs, metric='cosine')
            self._distance_matrix[vecname] = dict(leftids=leftids, rightids=rightids, sims=sims)
            
        if include_index:
            return (self._distance_matrix[vecname]['leftids'],
                    self._distance_matrix[vecname]['leftids'],
                    self._distance_matrix[vecname]['sims']
                   )
        else:
            return self._distance_matrix[vecname]['sims']
        
    def stat_pagecounts(self):
        lpc = self.left.meta(dedupe=True)['page_count']
        rpc = self.right.meta(dedupe=True)['page_count']
        return dict(leftpagecount=lpc,
                    rightpagecount=rpc,
                    pageDiff=lpc-rpc,
                    pagePropDiff=(lpc-rpc)/lpc
                    )
    
    def stat_simmat(self):
        simstats = dict()
        sim = self.distance_matrix()

        # For axis: Left is 1, Right is 0
        simstats['LeftSize'], simstats['RightSize'] = sim.shape
        simstats['minSize'] = min(sim.shape)
        left_min_margins = sim.min(axis=1) #most similar right match for each left page
        right_min_margins = sim.min(axis=0) # Unless looking for HTID to be reciprocal, may be unnecessary

        simstats['LeftMeanMinSim'] = left_min_margins.mean()
        simstats['RightMeanMinSim'] = right_min_margins.mean()
        simstats['MeanSim'] = sim.mean()

        # Only compare the X most similar numbers, where X is the size of the smaller margin
        if simstats['LeftSize'] > simstats['RightSize']:
            simstats['LeftTruncSim']= np.sort(left_min_margins)[:simstats['minSize']].mean()
            simstats['RightTruncSim'] = simstats['RightMeanMinSim']
        elif simstats['LeftSize'] < simstats['RightSize']:
            simstats['RightTruncSim']= np.sort(right_min_margins)[:simstats['minSize']].mean()
            simstats['LeftTruncSim'] = simstats['LeftMeanMinSim']
        else:
            simstats['RightTruncSim'] = simstats['RightMeanMinSim']
            simstats['LeftTruncSim'] = simstats['LeftMeanMinSim']

        threshold = 0.2
        # What proportion of left pages have a matching right page with a greater similarity (i.e. lower value) than `threshold`?
        for threshold in [0.005, 0.01, 0.03, 0.05]:
            simstats["LeftPropThresh{:03.0f}".format(threshold*100)] = left_min_margins[left_min_margins < threshold].shape[0] / left_min_margins.shape[0]
            
        for threshold in [0.005, 0.01, 0.03, 0.05, 0.08]:
            simstats["RightPropThresh{:03.0f}".format(threshold*100)] = right_min_margins[right_min_margins < threshold].shape[0] / right_min_margins.shape[0]
        
        return simstats
    
    def stat_sw(self, thresholds=[0.990, 0.995, 0.999]):
        ''' Thresholds differ based on vector approach used'''
        swstats = dict()
        for threshold in thresholds:
            self.assemble_sw_runs(threshold=threshold)
            swstats['SW{:04.0f}Len'.format(1000-threshold*1000)] = self.runs.shape[0]
        return swstats
    
    def all_stats(self):
        allstats = dict()
        for stat in ['pagecounts', 'sw', 'simmat']:
            quals = getattr(self, 'stat_' + stat)()
            allstats.update(quals)
        allstats.update(dict(left=self.left.htid, right=self.right.htid))
        return allstats
    
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