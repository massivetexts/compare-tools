import pandas as pd
import numpy as np
import logging

# match func should take an htid and return a dataframe with the following columns:
def full_ground_truth(left_htid, ground_truth, meta):
    ''' Add Same-author ground truth'''
    left_meta = meta[left_htid]
    left_gt = ground_truth[ground_truth.left == left_htid]
    author_htids = meta.get_where('author == "{}"'.format(left_meta['author']), ['htid']).htid
    author_htids = set(author_htids).difference(left_gt.right).difference({left_htid})
    author_gt = pd.DataFrame([dict(left=left_htid, right=htid, judgment='author') for htid in author_htids])
    return pd.concat([left_gt, author_gt])

def n_match_generator(left, ann, n, max_dist=None, min_count=None):
    df = ann.doc_match_stats(left, n=n, max_dist=max_dist, min_count=min_count)
    df = df.rename(columns={'target':'left', 'match': 'right'})[['left', 'right'] + list(df.columns[2:])].sort_values('mean')
    df['rank'] = np.arange(1, df.shape[0]+1)
    return df

def results_runner(left, match_func, grouth_truth, meta, include_fp=False):
    left_gt = full_ground_truth(left, grouth_truth, meta)
    df = match_func(left)
    results = left_gt.merge(df, how='left', on=['left','right']).sort_values('mean')
    if include_fp:
        false_positives = len(set(df.right).difference(left_gt.right).difference({left}))
        return results, false_positives
    else:
        return results

def run_eval(htids, condition_name, match_func, gt, meta, just_stats=False, print_every=1000):
    collector = []
    for i, left in enumerate(htids):
        try:
            if i % print_every == 0:
                print(i)
            if just_stats:
                results_raw, false_positives = results_runner(left, match_func, gt, meta, include_fp=True)
                false_negatives = results_raw.groupby(['judgment'])['rank'].apply(lambda x: x.isna().sum())
                true_positives = results_raw.groupby(['judgment'])['rank'].apply(lambda x: (~x.isna()).sum())
                recall_stats = pd.DataFrame([false_negatives, true_positives]).T.reset_index()
                recall_stats.columns = ['judgment', 'false_negatives', 'true_positives']
                recall_stats['left'] = left
                recall_stats['condition'] = condition_name
                false_positive_info = (left, condition_name, false_positives)
                collector.append((recall_stats, false_positive_info))
            else:
                results = results_runner(left, match_func, gt, meta)
                results['condition'] = condition_name
                collector.append(results)
        except:
            logging.warning('Issue with target ' + left)
            continue
    
    if just_stats:
        tp_fn = pd.concat(list(zip(*collector))[0])
        fp = pd.DataFrame(list(list(zip(*collector))[1]), columns=['left','condition', 'false_positives'])
        return tp_fn, fp
    else:
        return pd.concat(collector)

def main():
    from compare_tools.MTAnnoy import MTAnnoy
    from compare_tools.hathimeta import HathiMeta
    from compare_tools.configuration import config
    import argparse
    import time
    import os

    parser = argparse.ArgumentParser(description='Crunch ANNOY eval for a specific condition')
    parser.add_argument('--outdir', default='/data/saddl/annoyeval', type=str,
                        help='Directory to save the file of statistics to.')
    parser.add_argument('--filename', default="annoy-{}".format(time.time()), type=str,
                        help='Name of savefile. Is appended a tp_fn or fp (for true/false positives/negatives) and a parquet extension.')
    parser.add_argument('--ann-location', '-a', type=str,
                        help='Path to Annoy index.')
    parser.add_argument('--ann-dims', '-d', type=int, default=300, help='Dimensions of annoy index.')
    parser.add_argument('--ann-name', type=str, help='Name for the given ANN index.')
    parser.add_argument('--results-per-chunk', '-n', type=int, default=100,
                        help='Number of result chunks per target chunk.')
    parser.add_argument('--max-sim', '-m', type=float, default=1.,
                        help='Maximum cut-off similarity for inclusion in search results.')
    parser.add_argument('--min-count', '-c', type=int, default=1,
                        help='Minimum number of matching chunks between left and right.')
    #parser.add_argument('target_id_path', type=str, help="File with list of target ids to evaluation.")
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    
    # Args
    outpath = os.path.join(args.outdir, args.filename)
    
    print('Loading Ground Truth')
    gt = pd.read_parquet('/projects/saddl-main/sampling/ground_truth_meta_judgments.parquet')
    gt = gt[~gt.judgment.isin(['AUTHOR', 'DIFF'])] # Both of these categories are randomly sampled, so not useful here
    targets = gt.left.unique()
    if args.debug:
        targets = targets[:100]
    meta = HathiMeta(config['metadb_path'])
    
    print('Loading Annoy')
    ann = MTAnnoy(args.ann_location, dims=args.ann_dims)
    
    name = "max{}_n{}_c{}_ann{}".format(args.max_sim,
                                    args.results_per_chunk, 
                                    args.min_count,
                                    args.ann_name if args.ann_name else args.ann_location)
    print('Running Evaluation {} for {} ids'.format(name, len(targets)))
    matcher = lambda x: n_match_generator(x, ann, args.results_per_chunk, args.max_sim, args.min_count)
    tp_fn, fp = run_eval(targets, name, matcher, gt, meta, print_every=20, just_stats=True)

    print("Saving Results")
    tp_fn.to_parquet(outpath + '.tp_fn.parquet')
    fp.to_parquet(outpath + '.fp.parquet')
    
if __name__ == '__main__':
    main()