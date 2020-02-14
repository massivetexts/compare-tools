import numpy as np
import pandas as pd
import altair as alt
from collections import Counter
alt.data_transformers.enable('json')
from .utils import HTID

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

    def build_smith_waterman_matrix(self, threshold, metric, **kwargs):
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
        dm = self.similarity_matrix(metric = metric, **kwargs)
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
    
    def yield_sw_runs(self, threshold, no_joined_chunks = True, **kwargs):
        self.sw_copy = np.copy(self.sw_scores)
        while True:
            next_up = self.return_top_sw_sequence_and_remove_from_matrix(threshold, **kwargs)
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

    def assemble_sw_runs(self, mat = "similarity_matrix", threshold = 0.05, min_length = 3, min_strength = 0, metric = 'cosine', **kwargs):
        
        self.build_smith_waterman_matrix(threshold, metric = metric, **kwargs)
        self.sw_copy = np.copy(self.sw_scores)
        self.similarity_copy = np.copy(self.similarity_matrix(metric = metric, **kwargs))

        runs = []
        run_num = 0
        for run in self.yield_sw_runs(threshold, metric = metric, **kwargs):
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
        
    def return_top_sw_sequence_and_remove_from_matrix(self, threshold, **kwargs):
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
        mat = self.similarity_matrix(**kwargs)
        
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

    def jaccard_matrix(self, scope = 'page', drop_common = 200):
        """
        Return pairwise jaccard similarities across pages.

        Drops the 200 most common tokens.

        'document': The column in the initialized dataframe containing page-level info. 
        """
    
        if hasattr(self, "_jaccard_mat"):
            try:
                return self._jaccard_mat[scope]
            except:
                pass
        else:
            self._jaccard_mat = {}
        output = []
        lset, rset = self.left.tokensets(scope), self.right.tokensets(scope)
        mat = np.full((len(lset), len(rset)), np.nan)


        # Remove words from docs that are among the 200 most common
        counts = Counter()
        for myset in [lset, rset]:
            for section in myset:
                for word in section:
                    counts[word] += 1
                    
        dropping = set([word for word, count in counts.most_common(drop_common)])
        
        for myset in [lset, rset]:
            for section in myset:
                section -= dropping

                
        for i, (left_id, left_set) in enumerate(self.left.tokensets(scope).iteritems()):
            # Cache the left length
            left_length = len(left_set)
            for j, (right_id, right_set) in enumerate(self.right.tokensets(scope).iteritems()):
                if True: # i <= j:
                    inter = len(left_set.intersection(right_set))
                    t = (len(right_set) + left_length - inter)
                    if t == 0:
                        t = 1
                    sim = inter/t
                    mat[i, j] = sim
#                    mat[j, i] = sim
                    
        self._jaccard_mat[scope] = mat
#        self.similarity_matrix = mat
        return mat

    def plot(self, which = None, ids = None, runs = None, filter = None, width = 500,
             height = 500, scale_domain= None, **kwargs):
        '''
        
        scale_domain: an optional 'domain' argument to Altair's color scale. Use 'unaggregated'
        to size to the data.
        '''
        
        if which is None:
            which = "similarity_matrix"
        
        mat = getattr(self, which)(**kwargs)

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
        if runs:
            run_chart = alt.Chart(self.runs).encode(
                x = "left_seq:O",
                y = "right_seq:O",
                detail = 'run_num',
                tooltip = 'sim'
            ).mark_line(color='red')
            return grid + run_chart
            charts.append(run_chart)
            
        return alt.layer(*charts).properties(height=600, width=600)

    def jaccard_stats(self, scope):

        # Needed to initialize the matrix.
        _ = self.jaccard_matrix(scope = scope)
        self.runs = None

        self.assemble_sw_runs(threshold=.2, scope = scope, metric = 'jaccard')
        m = self.jaccard_matrix(scope = scope)

        jaccard_scores = dict(
            mean_page_diff = (self.runs.left_seq - self.runs.right_seq).abs().mean()
            , sd_page_diff = (self.runs.left_seq - self.runs.right_seq).abs().std()
            , mean_similarity_inside_runs = self.runs.sim.mean()
            , mean_similarity_to_right_of_runs = np.mean(m[self.runs.left_seq, self.runs.right_seq - 1])
            , share_l_in_r = len(self.runs.right_seq.unique())/self.jaccard_matrix(scope = scope).shape[1]
            , share_r_in_l = len(self.runs.left_seq.unique())/self.jaccard_matrix(scope = scope).shape[0]
        )
        if scope=='page':
            jaccard_scores.update(self.slopes())
            jaccard_scores.update(self.four_points())
        return jaccard_scores

    def slopes(self):
        """
        Get the slopes of run lines both looking in wordcount space (wheere SWDE should be equivalent unless there are footnotes) and
        in pagecount space (where same manifestations should always hes ave a slope of 1, and variation from that indicates definite 
        different pagination.

        This code is a pain because

        1. I hate hate hate pandas indexes. Just hate them. Indexes and columns should be the same thing. I hate pandas and python so much compared to R.
           How does pandas manage to be both far more verbose than R and also far less clear?
        2. More seriously, because there are multiple runs, and we need a sound way to aggregate across them.
           So I take the average slope within runs, and weight by the number of words or pages in the shorter of left or right.

        Currently, this will really only work on page-level counts.
        """
        z = self.left.tokenlist('page').reset_index().groupby('page').sum().cumsum().rename(columns={'count': 'left_count'})
        z.index = z.index.rename('left_seq')

        y = self.right.tokenlist('page').reset_index().groupby('page').sum().cumsum().rename(columns={'count': 'right_count'})
        y.index = y.index.rename('right_seq')

        runs = self.runs.reset_index().set_index('left_seq').join(z).reset_index().set_index('right_seq').join(y).reset_index()
        #page_slope = (runs.left_count[0] - self.runs.left_seq.values[-1])/(self.runs.right_seq[0] - self.runs.right_seq.values[-1])
        grouped = runs.groupby("run_num").agg(['min', 'max'])
        deltas = []
        for metric in ['seq', 'count']:
            delta = []
            for which in ['left', 'right']:
                e = grouped[f"{which}_{metric}"].copy()
                e['which'] = which
                e['metric'] = metric
                e['range'] = e['max'] - e['min']
                deltas.append(e)
        #    slopes['run_' + metric + '_slope'] = (delta[0]/delta[1])[0]

        all = pd.concat(deltas).reset_index().drop(columns=['min', 'max']).set_index(['run_num', 'metric']).pivot(columns='which').reset_index()

        all['weights'] = all['range'][['left', 'right']].mean(axis=1)
        all['slope'] = all['range']['left']/all['range']['right']
        all['weighted_slope'] = all['slope'] * all['weights']
        summed = all.groupby('metric').agg('sum')
        summed['slope'] = summed['weighted_slope']/summed['weights']
        summed.loc['count']['slope'][0]

        from sklearn.linear_model import LinearRegression
        X = self.runs.left_seq.values.reshape(-1, 1)
        y = self.runs.right_seq
        reg = LinearRegression().fit(X, y)
        overall_slope = reg.coef_[0]
        
        return dict(
            overall_slope = overall_slope,
            strength_of_overall_fit= reg.score(X, y),
            weighted_in_run_page_slope = all.groupby('metric')['slope'].mean().loc['seq'],
            weighted_in_run_word_slope = all.groupby('metric')['slope'].mean().loc['count']
        )
    
    def four_points(self, raw_data = False):
        m = self.jaccard_matrix(scope='page')
        flat = m.flatten() * -1
        shorter = min(*m.shape)
        flat.sort()
        flat *= -1
        flat = flat[:shorter*4]
        if raw_data:
            return flat
        return dict(
            sim_at_quarter_length = flat[shorter//4]
            , sim_at_half_length = flat[shorter//2]
            , sim_at_full_length = flat[shorter//1]
            , sim_at_double_length = flat[shorter*2]
        )


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
    
    def __init__(self, left = None, right = None, labels = ['left', 'right'], ids = None, **kwargs):
        if ids is not None:
            assert(len(ids)==2)
            self.left = HTID(ids[0], **kwargs)
            self.right = HTID(ids[1], **kwargs)
        else:
            self.left = left
            self.right = right
        self._distance_matrix = dict()

    def _repr_html_(self):
        return f"""<em>Comparison object</em> between <ol>
        <li>{self.left._repr_html_()}</li>
        <li>{self.right._repr_html_()}</li>
        </ol>
        """
    
    def similarity_matrix(self, vecname=None, include_index=False, metric='cosine', **kwargs):
        '''
        Vecname refers to the name of the Vector_file in HTID. If unnamed, HTID calls
        it 'vectors'. 'glove' and 'srp' are used for the SaDDL project.
        
        Uses 1 - distance matrix as a shortcut.
        '''
        if metric == 'cosine':
            dist = self.distance_matrix(vecname, include_index, **kwargs)
        if metric == 'jaccard':
            return self.jaccard_matrix(**kwargs)
        return 1 - dist
        
    def distance_matrix(self, vecname=None, include_index=False, **kwargs):
        '''
        Vecname refers to the name of the Vector_file in HTID. If unnamed, HTID calls
        it 'vectors'. 'glove' and 'srp' are used for the SaDDL project.
        '''
        from scipy.spatial.distance import cdist
        
        if not vecname:
            # Choose the name of first vector_file format given to left
            # e.g. if there is 'glove' and 'srp' in that order, choose 'glove'
            # (better to be explicit, of course)
            vecname = self.left._vecfiles[0][0]

        if (vecname not in self._distance_matrix) or not self._distance_matrix[vecname]:
            leftids, leftvecs = self.left.vectors(vecname)
            rightids, rightvecs = self.right.vectors(vecname)
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
    
    def stat_simmat(self, vecname='glove', thresholds=[0.002, 0.005, 0.01, 0.03, 0.05]):
        simstats = dict()
        sim = self.distance_matrix(vecname)

        # For axis: Left is 1, Right is 0
        simstats['LSize'], simstats['RSize'] = sim.shape
        simstats['minSize'] = min(sim.shape)
        simstats[vecname+'MeanSim'] = sim.mean()
        
        for side, axis in [('L', 1), ('R', 0)]:
            prefix = vecname + side
            min_margins = sim.min(axis=axis) #most similar right match for each left page
            
            simstats[prefix+'MeanMinSim'] = min_margins.mean()
            
            # Only compare the X most similar numbers, where X is the size of the smaller margin
            if simstats[side+'Size'] > simstats['minSize']: 
                simstats[prefix+'TruncSim']= np.sort(min_margins)[:simstats['minSize']].mean()
            else:
                simstats[prefix+'TruncSim'] = simstats[prefix+'MeanMinSim']

            # What proportion of left pages have a matching right page
            # with a greater similarity (i.e. lower value) than `threshold`?
            for threshold in thresholds:
                name = "{}PropDist{:04.0f}".format(prefix, threshold*1000)
                simstats[name] = min_margins[min_margins < threshold].shape[0] / min_margins.shape[0]
        
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
        for stat in []: #['pagecounts', 'sw']:
            quals = getattr(self, 'stat_' + stat)()
            allstats.update(quals)
        
        quals = self.stat_sw(thresholds=[0.990, 0.995, 0.996, 0.999])
        allstats.update(quals)
        
        quals = self.stat_simmat('glove', thresholds=[0.002, 0.005, 0.01, 0.02, 0.03])
        allstats.update(quals)
        #quals = self.stat_simmat('srp', thresholds=[0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.99])
        #allstats.update(quals)
        
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

#        self.similarity_matrix = mat
