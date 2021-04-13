from compare_tools.utils import HTID
from compare_tools.comparison import HTIDComparison
from compare_tools.MTAnnoy import MTAnnoy
from htrc_features import utils
import os
import time

class Saddler():
    '''
    The connector for putting the different moving parts together for SADDL - approximate nearest neighbour with MTAnnoy,
    pairwise Hathitrust book comparison with the HTID class, inference with a tensorflow model, and export to a JSON-based
    relationship format.
    '''
    
    def __init__(self, ann_path=None, ann_dims=300, prefault=False, htid_args={}):
        # If ann_path is empty, attempts to load from config['ann_path'], else config['full']['ann_path'] or config['test']['ann_path']
        if not ann_path:
            from compare_tools.configuration import config
            if 'ann_path' in config:
                ann_path = config['ann_path']
            elif 'full' in config and 'ann_path' in config['full']:
                ann_path = config['full']['ann_path']
            elif 'test' in config and 'ann_path' in config['test']:
                ann_path = config['test']['ann_path']
            else:
                raise Exception('No config ann_path location found. Specify it explicitly.')
        print(ann_path, ann_dims)
        self.mtannoy = MTAnnoy(ann_path, dims=ann_dims, prefault=prefault)
        self.htid_args = htid_args
    
    def get_candidates(self, htid, n=300, min_count=3, max_dist=.25, search_k=-1, save_to=None):
        '''
        save_to: location to save parquet output.
        '''
        results = self.mtannoy.doc_match_stats(htid, n=n, min_count=min_count, max_dist=max_dist, search_k=search_k)
        if save_to:
            os.makedirs(os.path.split(save_to)[0], exist_ok=True) # Create directories if needed
            results.to_parquet(save_to, compression='snappy')
        return results

    def get_simmats(self, htid, reshape=False, ann_args=dict(n=300, min_count=3, max_dist=.25)):
        '''
        For a target HTID, find match candidates with ANN, then return the target-candidate similarity matrices. Sim mats are unrolled, unless reshape argument is given. 
        '''
        left = HTID(htid, **self.htid_args)
        results = self.get_candidates(htid, **ann_args)

        rightindex = []
        allsims = []
        for i, rightid in enumerate(results.match.unique()):
            print(rightid)
            if htid == rightid:
                continue
            right = HTID(rightid, **hargs)
            comp = HTIDComparison(left, right)
            try:
                sims = comp.unrolled_sim()
                if reshape:
                    sims = sims.reshape(reshape)
                rightindex.append(rightid)
                allsims.append(sims)
            except:
                continue
        allsims = np.stack(allsims)
        return rightindex, allsims
    
    
def main():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument('--ann-path', type=str, default=None,
                        help="Location of MTAnnoy index. Default is None, which tries to fall back on what's in the config file")
    parser.add_argument('--ann-dims', type=int, default=50,
                        help='Number of dimensions for the MTAnnoy index.')
    parser.add_argument("--data-root", type=str, default='/data/saddl/full/',
                        help="Location to save stubbytree data file outputs")
    ann_parser = subparsers.add_parser("Candidates",
                                       help="Save candidate relationships from ANN")
    #b_parser = subparsers.add_parser("B")

    # Configure for the MTAnnoy candidate retrieval
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
    
    if not args.ann_path:
        from compare_tools.configuration import config
        if 'ann_path' in config:
            ann_path = config['ann_path']
        elif 'full' in config and 'ann_path' in config['full']:
            ann_path = config['full']['ann_path']
        elif 'test' in config and 'ann_path' in config['test']:
            ann_path = config['test']['ann_path']
        else:
            raise Exception('No config ann_path location found. Specify it explicitly.')
    else:
        ann_path = args.ann_path
    
    if not args.command:
        parser.print_help()
        return
    
    saddlr = Saddler(ann_dims=args.ann_dims, ann_path=ann_path, prefault=False)
    
    if args.command == 'Candidates':
        if args.htid_in:
            htids = [htid.strip() for htid in args.htid_in]
        else:
            htids = args.htids
        
        starttime = time.time()
        skipped = 0
        for i, htid in enumerate(htids):
            try:
                outpath = os.path.join(args.data_root, utils.id_to_stubbytree(htid, format='ann.parquet'))
                if os.path.exists(outpath) and not args.overwrite:
                    print('skipping already processed: {}'.format(outpath))
                    skipped += 1
                    continue

                results = saddlr.get_candidates(htid,
                                                n=args.results_per_chunk, 
                                                min_count=args.min_count, 
                                                max_dist=args.max_dist,
                                                search_k=args.search_k,
                                                save_to=outpath)
                
                if i % 1 == 0:
                    progress = (time.time() - starttime)/60
                    remaining = progress/(i-skipped) * (len(htids)-i-skipped)
                    print(f"{i-skipped}/{len(htids)-skipped} completed in {progress:.1f}min (Est left: {remaining:.1f}min)")
            except KeyboardInterrupt:
                raise
            
            except:
                print("Issue with {}".format(htid))

if __name__ == '__main__':
    main()
