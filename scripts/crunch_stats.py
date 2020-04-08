from compare_tools.hathimeta import HathiMeta, get_json_meta
from compare_tools.configuration import config, init_htid_args
from compare_tools.comparison import Comparison, HTIDComparison
from compare_tools.utils import HTID
import time
import os
import pandas as pd
import argparse
import json

def main():
    htid_args = init_htid_args(config)

    parser = argparse.ArgumentParser(description='Take a CSV of left/right judgments to crunch statistics for.')
    parser.add_argument('--outdir', default='/data/saddl/stats', type=str,
                        help='Directory to save the file of statistics to.')
    parser.add_argument('--filename', default="stats-{}.parquet".format(time.time()), type=str,
                        help='Name of savefile.')
    parser.add_argument('input_dicts', nargs='+', type=str, 
                        help="dicts of format {'left':'x', 'right': 'y'} to process. If you're splitting up external data, it's "
                        "more efficient to sort by 'left' columns first")

    args = parser.parse_args()

    outpath = os.path.join(args.outdir, args.filename)
    stats_collector = []
    i = 0 
    start = time.time()

    injson = [json.loads(d) for d in args.input_dicts]
    df = pd.DataFrame(injson)

    print(len(args.input_dicts))
    for leftid, matches in df.groupby('left'):
        if i % 1000 == 0:
            print(i, len(stats_collector), leftid, (time.time()-start)/60)
        left = HTID(leftid, **htid_args)

        for rightid in matches['right']:
            right = HTID(rightid, **htid_args)
            comp = HTIDComparison(left, right)
            try:
                stats = comp.all_stats()
                stats.update(comp.stat_quantiles('srp'))
                stats.update(comp.stat_quantiles('glove'))
                stats_collector.append(stats)
            except KeyboardInterrupt:
                raise
            except:
                print("Issue with ", left.htid, right.htid)
                continue
        i += 1

    pd.DataFrame(stats_collector).to_parquet(outpath, compression='snappy')
    print((time.time() - start)/60)
    
if __name__ == '__main__':
    main()