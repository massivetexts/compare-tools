from .utils import htid_ize
import pandas as pd
import SRP
from collections import defaultdict
import numpy as np
import random
from .configuration import meta_path

"""
Utilities for working with a collection of chunks.
"""

def random_mtid(frame, n):
    return(frame.sample(n).reset_index()['mtid'])


def expand_mtid(df, label):
    # spread out an mtid into htid and section with a prefix
    x = df[label].str.rsplit("-", 1, expand = True)
    x.columns = [label+'_htid', label+'_section']
    return pd.concat([df, x], 1)
    df[label+'_htid'] = [parts[0] for part in parts]
    df[label+'_chunk'] = [int(parts[1]) for part in parts]

def expand_left_right(data):
    # Provide htid and section labels.
    data = expand_mtid(data, 'left')
    data = expand_mtid(data, 'right') 
    return data    
    
class ChunkCollection():
    """
    This class holds an embedded set of chunks (SRP or Glove), 
    along with associated metadata

    """
    def __init__(self, chunk_file, metadata_file = meta_path):
        self.metadata = pd.read_csv(metadata_file, low_memory=False).set_index('htid')
        self._initialize_embeddings(chunk_file)
        
    def _initialize_embeddings(self, chunk_file):
        """
        Read in an embedding file at unit length, and adjust the metadata to match.
        """
        input = SRP.Vector_file(chunk_file)
        dataset = input.to_matrix(unit_length = True)
        self.matrix = dataset['matrix']
        
        ids = dataset['names']
        self.ids = ids

        sections = [a.split("-") for a in dataset['names']]
        try:
            htids, sections, starts, ends = zip(*sections)
        except:
            # The old format: deprecated.
            htids, sections = zip(*sections)
        sections = pd.DataFrame({'mtid': ids, 'htid': htids, 'section': list(map(int, sections))}).set_index('htid')

        self.chunk_metadata = sections.join(self.metadata, how='left').reset_index()
        
        self.mtid_lookup = dict(zip(dataset['names'], range(len(dataset['names']))))
        
        self.htid_lookup = defaultdict(list)
        for i, htid in enumerate(htids):
            self.htid_lookup[htid].append(ids[i])

    def get_named_result_df(self, mtid = None, htid = None, n=30, comps_per_item = 31):
        if comps_per_item <= n:
            # Make sure you actually get the guaranteed number of comparisons.
            comps_per_item = n+1
            
        if mtid is not None:
            d= self.get_nns_by_mtid(mtid, comps_per_item)
            
        if htid is not None:
            d = self.get_nns_by_htid(htid, comps_per_item)
            
        data = pd.DataFrame(d, columns = ["left", "right", "sim"])
        data = data[data['left'] != data['right']]
        d = data["left"].str.split("-", n=1, expand = True)
        data["target"] = d[0]
        data["target_seq"] = d[1].astype('int')

        d = data["right"].str.split("-", n=1, expand = True)
        data["match"] = d[0]
        data["match_seq"] = d[1].astype('int')

        # Don't self-match.
#        data = data[data['match'] != data['target']]
        
        # Dropping old Name columns
        data.drop(columns =["left", "right"], inplace = True)
        data = data.sort_values('sim', ascending = False)
        return data
#        return data.head(n)
    
    def get_nns_by_mtid(self, id, comps_per_item = 30):
        left_set = np.array([self.matrix[self.mtid_lookup[id]]])
        closest = self.closest(left_set, comps_per_item)
        return [(id, neighbor, sim) for (ix, neighbor, sim) in closest]
    
    def get_nns_by_htid(self, htid, comps_per_item = 30):
        ids = [id for id in self.htid_lookup[htid]]
        left_set = np.array([self.matrix[self.mtid_lookup[id]] for id in ids])
        closest = self.closest(left_set, comps_per_item)
        return [(ids[ix], neighbor, sim) for (ix, neighbor, sim) in closest]
    
    def closest(self, left_set, comps_per_item):
        ds = np.dot(left_set, self.matrix.T)
        partitioned = np.argpartition(ds, -comps_per_item, axis=1)[...,-comps_per_item:]
        neighbors = []
        for i, row in enumerate(partitioned):
            for item in row:
                neighbors.append((i, self.ids[item], ds[i, item]))
        return neighbors

    def brute_cosine(self):
        distances = np.dot(self.vectorset, self.corpus.matrix.T)
        return distances

    def neighbors(self, comps_per_item = 10, minimum_matches = 5):
        ds = self.brute_cosine()
        partitioned = np.argpartition(ds, -comps_per_item, axis=1)[...,-comps_per_item:]


        arr = map(htid_ize, np.array(neighbor_names.flatten()))
        from collections import Counter
        commons = Counter(arr).most_common(10)
        candidates = [c for c in commons if c[1] > minimum_matches and c[0] != self.htid]
        return candidates        
        
        
    def mtid_matrix(self, mtids):
        """
        return only the n rows matching a given mtid as an nxk numpy matrix
        where k is the vector length.
        """
        output = np.zeros((len(mtids), self.matrix.shape[1]))
        for i, mtid in enumerate(mtids):
            output[i] = self.matrix[self.mtid_lookup[mtid]]
        return output

        
    def htid_matrix(self, htids):
        """
        yield a matrix with chunks for each htid in the htids AND
        the associated IDs.
        """
        output = []
        for id in htids:
            output.extend(self.htid_lookup[id])
        return (output, self.mtid_matrix(output))
    
    def paired_distance(self, left_ids, right_ids = None, adjusted = False, format = 'pandas'):
        """
        Given two sets of ids, find all pairwise distances between the left and the right set.
        If right ids is none, find self-similarity in a single set of ids.

        'ids' may be either mtids or htids; if htids, they will be expanded to mtids.
        
        'Adjusted' recenters the matrices around the median point (on each dimension) 
        in the passed dataset. This means that there should be no baseline 
        similarity when all the books are--say-- in English.

        format is either 'pandas': return a three column array with left id, right id, and sim;
          or 'matrix': return a tuple of ids and a numpy matrix.
        
        """
        if right_ids is None:
            # Default to self-distance
            right_ids = left_ids

        # Accept htids as well as mtids by checking the first element.
        def expand_htids(iset):
            if not '-' in list(iset)[0]:
                output = []
                for id in iset:
                    output.extend(self.htid_lookup[id])
                output.sort()
                return output
            return iset

        left_ids, right_ids = map(expand_htids, [left_ids, right_ids])
        
        left = self.mtid_matrix(left_ids)
        right = self.mtid_matrix(right_ids)
        if adjusted:
            joint = np.concatenate((left, right))
            mediod = np.median(joint, 0)
            left = left - mediod
            right = right - mediod
        ds = (np.dot(left, right.T))
        labs = [(ida, idb) for ida in left_ids  for idb in right_ids]
        if format == 'matrix':
            return (ds, (left_ids, right_ids))
        return pd.DataFrame([(a, b, c) for ((a, b), c) in (zip(labs, ds.flatten()))], columns = ['left', 'right', 'sim'])

        
    def random_paired_distance(self, element = "random", max = 100, adjusted = False):
        """
        Generate some random pairs of mtids from pd Dataframe with sections in it.
        Useful for just seeing what different types of comps look like.

        element must be a column name in the chunk metadata; it allows you to ask
        for random comparison only within a single author, within a single htid, etc..

        max is the maximum number of comparisons.

        'adjusted' is passed to paired distance.
        """
        sections = self.chunk_metadata
        
        if element == "random":
            left_ids = random_mtid(sections, max)
            right_ids = random_mtid(sections, max)
        else:
            key = random.sample(list(sections[element]), 1)[0]
            ids = set(sections[sections[element] == key].mtid)
            left_ids = ids
            right_ids = ids
            if len(ids) > max:
                left_ids = random.sample(list(ids), max)
                right_ids = random.sample(list(ids), max)
        out = self.paired_distance(left_ids, right_ids)
        out['relation'] = element
            
        return expand_left_right(out)





