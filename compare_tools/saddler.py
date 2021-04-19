from compare_tools.utils import HTID
from compare_tools.comparison import HTIDComparison
from compare_tools.MTAnnoy import MTAnnoy
from htrc_features import utils
import os
import time
import numpy as np
import pandas as pd
import logging
# For Prediction
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from compare_tools.train_utils import judgment_labels, judgment_label_ref
from tensorflow.keras.metrics import top_k_categorical_accuracy

def top_2_accuracy(x, y):
  return top_k_categorical_accuracy(x, y, k=2)

class Saddler():
    '''
    The connector for putting the different moving parts together for SADDL - approximate nearest neighbour with MTAnnoy,
    pairwise Hathitrust book comparison with the HTID class, inference with a tensorflow model, and export to a JSON-based
    relationship format.
    '''
    
    def __init__(self, htid_args={}, data_dir=None):
        self._mtannoy = None
        
        # Load config file for params.
        try:
            from compare_tools.configuration import config
            config.update(config['test'])
            config.update(config['full'])
            self.config = config
        except:
            self.config = {}
        
        try:
            if not data_dir:
                self.data_dir = config['data_dir']
            else:
                self.data_dir = data_dir
        except:
            raise Exception("If the following args are not provided, they need to be in ~/.htrc_config.yaml: data_dir")

        self.htid_args = htid_args
    
    def mtannoy(self, ann_path=None, ann_dims=None, prefault=False, force=False):
        if self._mtannoy and not force:
            return self._mtannoy
        
        # Load
        if not ann_dims:
            ann_dims = self.config['ann_dims']
        
        if not ann_path:
            ann_path = self.config['ann_path']
            
        self._mtannoy = MTAnnoy(ann_path, dims=ann_dims, prefault=prefault)
        return self._mtannoy
    
    def get_candidates(self, htid, n=300, min_count=3, max_dist=.25, search_k=-1, force=False, 
                       save=False, ann_path=None, ann_dims=None, prefault=False):
        '''
        force: Force recrunch, even if the file already exists. Otherwise, existing file is loaded.
        save: save results to data_dir with stubbytree file structure
        '''
        outpath = os.path.join(self.data_dir, utils.id_to_stubbytree(htid, format='ann.parquet'))

        mtannoy = self.mtannoy(ann_path, ann_dims, prefault, force=False)
        
        if not force and os.path.exists(outpath):
            print('File already found: {}'.format(outpath))
            results = pd.read_parquet(outpath)
        else:
            results = mtannoy.doc_match_stats(htid, n=n, min_count=min_count, max_dist=max_dist, search_k=search_k)
            if save:
                os.makedirs(os.path.split(outpath)[0], exist_ok=True) # Create directories if needed
                results.to_parquet(outpath, compression='snappy')

        return results

    def get_simmats(self, htid, reshape=False, max_size=150, include_wem=True, 
                    ann_args=dict(n=300, min_count=3, max_dist=.25)):
        '''
        For a target HTID, find match candidates with ANN, then return the target-candidate similarity matrices. 
        Sim mats are unrolled, unless reshape argument is given. 
        '''
        results = self.get_candidates(htid, **ann_args)
        rightindex, allsims = self._get_simmats_from_candidates(htid, reshape,
                                                                max_size=max_size, 
                                                                include_wem=include_wem)
        return rightindex, allsims
    
    def _get_simmats_from_candidates(self, candidates, max_size=150, reshape=False, include_wem=True):
        '''
        include_wem: Also return an averaged vec for each of the two full books, concatenated.
        '''
        htid = candidates.target.iloc[0]
        left = HTID(htid, **self.htid_args)
        if include_wem:
            rightvecs = []
            try:
                leftvec = left.vectors('glove')[1].mean(0)
                assert not np.isnan(leftvec).any()
            except:
                logging.warning("Issue with left vec {}".format(left.htid))
                return None
        
        rightindex = []
        allsims = []
        for i, rightid in enumerate(candidates.match.unique()):
            if htid == rightid:
                continue
            right = HTID(rightid, **self.htid_args)
            comp = HTIDComparison(left, right)
            try:
                sims = comp.unrolled_sim(max_size=max_size)
                if reshape:
                    sims = sims.reshape(reshape)
                rightindex.append(rightid)
                allsims.append(sims)

                if include_wem:
                    rightvec = right.vectors('glove')[1].mean(0)
                    rightvecs.append(rightvec)
            except:
                logging.warning("Issue with {} v. {}".format(left.htid, right.htid))
                continue
        allsims = np.stack(allsims)
        if not include_wem:
            return rightindex, allsims
        else:
            rightvecs = np.vstack(rightvecs)
            wem_mean = np.mean([leftvec, rightvecs], axis=0)
            wem_diff = np.subtract(leftvec, rightvecs)
            wem_concat = np.hstack([wem_mean, wem_diff])
            return rightindex, (allsims, wem_concat)
    
def main():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument("--data-root", type=str, default='/data/saddl/full/',
                        help="Location to save stubbytree data file outputs")
    ann_parser = subparsers.add_parser("Candidates",
                                       help="Save candidate relationships from ANN")
    #b_parser = subparsers.add_parser("B")

    # Configure for the MTAnnoy candidate retrieval
    ann_parser.add_argument('--ann-path', type=str, default=None,
                            help="Location of MTAnnoy index. Default is None, which tries to fall " \
                            "back on what's in the config file")
    ann_parser.add_argument('--ann-dims', type=int, default=50,
                            help='Number of dimensions for the MTAnnoy index.')
    ann_parser.add_argument('--prefault', action='store_true', help='Load ANN into memory.')
    ann_parser.add_argument("--results-per-chunk", "-n", type=int, default=300,
                            help="Number of ANN results to return per chunk")
    ann_parser.add_argument("--min-count", type=int, default=2,
                            help="Min number of matching chunks between books.")
    ann_parser.add_argument("--search-k", type=int, default=-1,
                            help="ANN search k parameter.")
    ann_parser.add_argument("--max-dist", type=float, default=.25,
                            help="Maximum distance between matching chunks.")
    ann_parser.add_argument("--overwrite", action="store_true",
                            help="Overwrite files if they already exist. Otherwise, they're skipped")
    ann_parser.add_argument("--htid-in", type=argparse.FileType('r'), default=None, help='File of HTIDs to process. If set, htids args provided on the command line are ignored.')
    ann_parser.add_argument("htids", nargs='*', help='HTIDs to process. Alternately, provide --htid-in')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    saddlr = Saddler(data_dir=args.data_root)
    
    if args.command == 'Candidates':
        # Pre-load MTAnnoy. Unnecessary, but more readable
        saddlr.mtannoy(ann_dims=args.ann_dims, ann_path=args.ann_path, prefault=args.prefault)
        
        if args.htid_in:
            htids = [htid.strip() for htid in args.htid_in]
        else:
            htids = args.htids
        
        starttime = time.time()
        skipped = 0
        for i, htid in enumerate(htids):
            try:
                results = saddlr.get_candidates(htid,
                                                n=args.results_per_chunk, 
                                                min_count=args.min_count, 
                                                max_dist=args.max_dist,
                                                search_k=args.search_k,
                                                force=args.overwrite,
                                                save=True)
                
                if i % 1 == 0:
                    progress = (time.time() - starttime)/60
                    remaining = progress/(i-skipped) * (len(htids)-i-skipped)
                    print(f"{i-skipped}/{len(htids)-skipped} completed in {progress:.1f}min (Est left: {remaining:.1f}min)")
            except KeyboardInterrupt:
                raise
            
            except:
                raise
                print("Issue with {}".format(htid))

if __name__ == '__main__':
    main()
