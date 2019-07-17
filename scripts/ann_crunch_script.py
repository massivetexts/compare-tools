from compare_tools.MTAnnoy import MTAnnoy, create_annoy_index
import pandas as pd
import argparse
import os
from htrc_features import utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--out-dir', default='matches/', type=str, nargs='+',
                    help='Directory to save the file to.')
parser.add_argument('htids', nargs='+', type=str, 
                    help='htids to process.')

args = parser.parse_args()

ann = MTAnnoy('testsetGlove-25trees.ann', 300)

all_dfs = []
for htid in args.htids:
    try:
        target_length = ann.ind.loc[htid].length
        df = ann.get_named_result_df(htid=htid, dedupe=True)
        stats = df.groupby(['target', 'match'])['dist'].aggregate(['count', 'mean']).sort_values('count', ascending=False).reset_index(0)
        stats = pd.merge(stats, ann.ind.loc[stats.index].length, right_index=True, left_index=True)
        stats['prop_target'] = stats['count'] / target_length
        stats['prop_match'] = stats['count'] / stats.length
        stats = stats.reset_index()
        
        # Testing has shown that comparisons with a greater mean distance than ~0.16 are likely 
        # not useful. To play it safe but avoid too much junk, filter where mean >0.3
        stats = stats[stats['mean'] <= 0.3]
        
        all_dfs.append(stats)
    except:
        with open('errors.txt', mode='a') as f:
            f.write(htid)
            
# Named after the first file in the set
path = os.path.join('matches', '%splus%d.parquet' % (utils.clean_htid(htid),len(args.htids)-1))
pd.concat(all_dfs).to_parquet(path, compression='snappy')