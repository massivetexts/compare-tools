from compare_tools.hathimeta import HathiMeta, get_json_meta
from compare_tools.configuration import config, init_htid_args
from compare_tools.comparison import Comparison, HTIDComparison
from compare_tools.utils import HTID
import logging
from compare_tools.train_utils import df_to_tfrecords, _serialize_series, judgment_label_ref
import time
import os
import pandas as pd
import argparse
import json
from io import StringIO

def main():
    logging.basicConfig(level=logging.INFO)
    config.update(config['full'])
    htid_args = init_htid_args(config)

    parser = argparse.ArgumentParser(description='Take a set of left/right judgments to crunch statistics for.')
    parser.add_argument('--outdir', default='/data/saddl/stats', type=str,
                        help='Directory to save the file of statistics to.')
    parser.add_argument('--filename', default="stats-{}".format(time.time()), type=str,
                        help='Name of savefile. Extension automatically added.')
    parser.add_argument('--tfrecord', action='store_true', help='Whether to save output as a TFRecord.')
    parser.add_argument('--save-sim', action='store_true',
                        help='Save an unrolled similarity matrix instead of the statistics.')
    parser.add_argument('--input-file', '-i', type=str,  help='Filepath for input data, to use instead of input_dicts.')
    parser.add_argument('--matrix-size', '-m', type=int, default=100,
                        help='Size of similarity matrix. Anything smaller is padded, anything longer is truncated.')
    parser.add_argument('--no-compress', action='store_true',  help='Avoid tfrecord compression.')
    parser.add_argument('input_dicts', nargs='*', type=str,
                        help="dicts of format {'left':'x', 'right': 'y'} to process. If you're splitting up external data, it's "
                        "more efficient to sort by 'left' columns first. If you're saving from a DataFrame with 'left' and 'right' "
                        "columns"", you can convert to the format with df.to_dict(orient='records'). You can also add 'judgment' and 'notes' "
                        "to the input data if exporting a TFRecord for training."
                       )

    args = parser.parse_args()
    
    if (len(args.input_dicts) == 0) and not args.input_file:
        raise Exception("Need either input_dicts or an input_file")
    elif (len(args.input_dicts) > 0) and args.input_file:
        raise Exception("Can't set both input_dicts and input_file")
    elif len(args.input_dicts) > 0:
        chunks = pd.read_json(StringIO("\n".join(args.input_dicts)),
                              orient='records', lines=True, chunksize=5000)
    elif args.input_file:
        chunks = pd.read_json(args.input_file, 
                              orient='records', lines=True, chunksize=5000)
    if args.tfrecord:
        import tensorflow as tf
        if not args.save_sim:
            raise Exception('TFRecord can only save a padded similarity matrix.')
        args.filename = args.filename + '.tfrecord'
        outpath = os.path.join(args.outdir, args.filename)
        options = tf.io.TFRecordOptions(compression_type="GZIP" if not args.no_compress else "")
        writer = tf.io.TFRecordWriter(outpath, options=options)
    else:
        args.filename = args.filename + '.parquet'
        outpath = os.path.join(args.outdir, args.filename)

    stats_collector = []
    n = 0 
    start = time.time()

    for df in chunks:
        for leftid, matches in df.groupby('left'):
            left = HTID(leftid, **htid_args)
            for j, match in matches.iterrows():
                if n % 1000 == 0:
                    logging.info("{}, {}, {}, {}".format(n, len(stats_collector), leftid, (time.time()-start)/60))
                rightid = match['right']
                right = HTID(rightid, **htid_args)
                comp = HTIDComparison(left, right)
                try:
                    if args.save_sim:
                        vec = comp.unrolled_sim(max_size=args.matrix_size)
                        stats = dict(zip([str(k) for k in range(vec.shape[0])], vec))
                        stats['left'] = leftid
                        stats['right'] = rightid
                        if args.tfrecord:
                            stats['judgment'] = match['judgment'] if 'judgment' in match else 'UNKNOWN'
                            stats['notes'] = str(match['notes']) if 'notes' in match else ''
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
                n += 1

        if args.tfrecord:
            out = pd.DataFrame(stats_collector)
            for i, row in out.iterrows():
                serialized = _serialize_series(row, judgment_label_ref)
                writer.write(serialized)
            stats_collector = []
        else:
            # parquet doesn't support append, so stats_collector will keep growing until the end.
            # careful with big batches
            pass

    if args.tfrecord:
        writer.close()
    else:
        out = pd.DataFrame(stats_collector)
        out.to_parquet(outpath, compression='snappy')
        
    
    logging.info("{} records parsed in {} min".format(n, ((time.time() - start)/60)))
    
if __name__ == '__main__':
    main()