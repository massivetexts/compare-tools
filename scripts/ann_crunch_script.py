from compare_tools.MTAnnoy import MTAnnoy, create_annoy_index
import pandas as pd
import argparse
import os
from htrc_features import utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--out-dir', default='matches/', type=str,
                    help='Directory to save the file to.')
parser.add_argument('--min-mean', default=0.3, type=float,
                    help='Minimum mean similarity.')
parser.add_argument('-n', '--matches-per-chunk', default=30, type=int,
                    help='Number of matches per chunk.')
parser.add_argument('htids', nargs='+', type=str, 
                    help='htids to process.')

args = parser.parse_args()

ann = MTAnnoy('../../ann-ef/testsetGlove-25trees.ann', 300)

all_dfs = []

for htid in args.htids:
    try:
        stats = ann.doc_match_stats(htid=htid, =args.matches_per_chunk)
        
        # Testing has shown that comparisons with a greater mean distance than ~0.16 are likely 
        # not useful. To play it safe but avoid too much junk, filter where mean >0.3
        stats = stats[stats['mean'] <= args.min_mean]
        
        all_dfs.append(stats)
    except:
        with open('errors.txt', mode='a') as f:
            f.write(htid)
            
# Named after the first file in the set
path = os.path.join(args.out_dir, '%splus%d.parquet' % (utils.clean_htid(htid),len(args.htids)-1))
pd.concat(all_dfs).to_parquet(path, compression='snappy')