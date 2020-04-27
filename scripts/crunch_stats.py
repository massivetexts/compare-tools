from compare_tools.hathimeta import HathiMeta, get_json_meta
from compare_tools.configuration import config, init_htid_args
from compare_tools.comparison import Comparison, HTIDComparison
from compare_tools.utils import HTID
import logging
from compare_tools.train_utils import df_to_tfrecords
import time
import os
import pandas as pd
import argparse
import json

def main():
    logging.basicConfig(level=logging.INFO)
    
    htid_args = init_htid_args(config)

    parser = argparse.ArgumentParser(description='Take a set of left/right judgments to crunch statistics for.')
    parser.add_argument('--outdir', default='/data/saddl/stats', type=str,
                        help='Directory to save the file of statistics to.')
    parser.add_argument('--filename', default="stats-{}".format(time.time()), type=str,
                        help='Name of savefile. Extension automatically added.')
    parser.add_argument('--tfrecord', action='store_true', help='Whether to save output as a TFRecord.')
    parser.add_argument('--save-sim', action='store_true',
                        help='Save an unrolled similarity matrix instead of the statistics.')
    parser.add_argument('input_dicts', nargs='+', type=str,
                        help="dicts of format {'left':'x', 'right': 'y'} to process. If you're splitting up external data, it's "
                        "more efficient to sort by 'left' columns first. If you're saving from a DataFrame with 'left' and 'right' "
                        "columns"", you can convert to the format with df.to_dict(orient='records'). You can also add 'judgment' and 'notes' "
                        "to the input data if exporting a TFRecord for training."
                       )

    args = parser.parse_args()

    if args.tfrecord:
        if not args.save_sim:
            raise Exception('TFRecord can only save a padded similarity matrix.')
        args.filename = args.filename + '.tfrecord'
    else:
        args.filename = args.filename + '.parquet'

    outpath = os.path.join(args.outdir, args.filename)
    stats_collector = []
    i = 0 
    start = time.time()

    injson = [json.loads(d) for d in args.input_dicts]
    df = pd.DataFrame(injson)

    print(len(args.input_dicts))
    for leftid, matches in df.groupby('left'):
        if i % 1000 == 0:
            logging.info("{}, {}, {}, {}".format(i, len(stats_collector), leftid, (time.time()-start)/60))
        left = HTID(leftid, **htid_args)

        for i, match in matches.iterrows():
            rightid = match['right']
            right = HTID(rightid, **htid_args)
            comp = HTIDComparison(left, right)
            try:
                if args.save_sim:
                    vec = comp.unrolled_sim()
                    stats = dict(zip([str(i) for i in range(vec.shape[0])], vec))
                    stats['left'] = leftid
                    stats['right'] = rightid
                    if args.tfrecord:
                        stats['judgment'] = match['judgment'] if 'judgment' in match else 'UNKNOWN'
                        stats['notes'] = match['notes'] if 'notes' in match else ''
                    stats_collector.append(stats)
                else:
                    stats = comp.all_stats()
                    stats.update(comp.stat_quantiles('srp'))
                    stats.update(comp.stat_quantiles('glove'))
                    stats_collector.append(stats)
            except KeyboardInterrupt:
                raise
            except:
                logging.warning("Issue with {} v. {}".format(left.htid, right.htid))
                continue
        i += 1

    df = pd.DataFrame(stats_collector)
    if args.tfrecord:
        df_to_tfrecords(df, outpath)
    else:
        df.to_parquet(outpath, compression='snappy')
    logging.info("{} records parsed in {}min".format(len(df), 
                                                   ((time.time() - start)/60)))
    
if __name__ == '__main__':
    main()