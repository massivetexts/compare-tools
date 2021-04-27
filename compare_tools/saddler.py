from compare_tools.utils import HTID
from compare_tools.comparison import HTIDComparison
from compare_tools.MTAnnoy import MTAnnoy, TitleAnnoy
from htrc_features import utils
import os
import time
import numpy as np
import pandas as pd
import logging
import rapidjson as json
import dask.dataframe as dd
from compare_tools.configuration import init_htid_args
# For Prediction
#import tensorflow as tf
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from compare_tools.train_utils import judgment_labels, judgment_label_ref

class Saddler():
    '''
    The connector for putting the different moving parts together for SADDL - approximate nearest neighbour with MTAnnoy,
    pairwise Hathitrust book comparison with the HTID class, inference with a tensorflow model, and export to a JSON-based
    relationship format.
    '''
    
    def __init__(self, htid_args=None, data_dir=None):
        self._mtannoy = None
        self._tf_model = None
        self._titleann = None
        
        # Load config file for params.
        try:
            from compare_tools.configuration import config
            if 'test' in config:
                config.update(config['test'])
            if 'full' in config:
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

        self._htid_args = htid_args
    
    @property
    def htid_args(self):
        ''' Args used by the HTID class (mainly pointers to various other classes used by 
        each HTID object). If empty, tries to load from config'''
        if not self._htid_args:
            self._htid_args = init_htid_args(self.config)
        return self._htid_args
            
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
    
    def tf_model(self, model_path, force=False):
        from tensorflow.keras.models import load_model
        from tensorflow.keras.metrics import top_k_categorical_accuracy
        
        if not model_path:
            if 'model_path' in self.config:
                model_path = self.config['model_path']
            else:
                raise Exception("Need to either specify a model path or include in config file")
    
        def top_2_accuracy(x, y):
            return top_k_categorical_accuracy(x, y, k=2)

        print("Loading model at", model_path)
        if not self._tf_model or force:
            self._tf_model = load_model(model_path, custom_objects={"top_2_accuracy":top_2_accuracy})
            
        return self._tf_model
    
    def get_candidates(self, htid, n=300, min_count=3, max_dist=.25, search_k=-1, force=False, 
                       save=False, ann_path=None, ann_dims=None, prefault=False):
        '''
        force: Force recrunch, even if the file already exists. Otherwise, existing file is loaded.
        save: save results to data_dir with stubbytree file structure
        '''
        outpath = os.path.join(self.data_dir, utils.id_to_stubbytree(htid, format='ann.parquet'))

        if not force and os.path.exists(outpath):
            print('File already found: {}'.format(outpath))
            results = pd.read_parquet(outpath)
        else:
            mtannoy = self.mtannoy(ann_path, ann_dims, prefault, force=False)
            results = mtannoy.doc_match_stats(htid, n=n, min_count=min_count, max_dist=max_dist, search_k=search_k)
            if save:
                os.makedirs(os.path.split(outpath)[0], exist_ok=True) # Create directories if needed
                results.to_parquet(outpath, compression='snappy')

        return results

    def get_predictions(self, htid, save_all=False, force_all=False,
                        save_candidates=False, save_predictions=False, save_output=False,
                        force_candidates=False, force_predictions=False, force_output=False,
                        skip_json_output=False,
                        ann_args=dict(n=300, min_count=3, max_dist=.25)):
        '''
        Front to back prediction, from getting candidates to crunching predictions to formatting as JSON dataset.
        
        Uses default arguments for many subprocesses, so if you need something more customized, run individually.
        
        save_all: Save intermediate and final data, to stubbytree format. If True, overwrites save_candidates,
                save_predictions, and save_output; if False, those more fine-turned options are used.
        force_all: Force processing for all steps, ignoring whether a file has already been saved.
        '''
        if save_all:
            save_candidates = True
            save_predictions = True
            save_output = True
        if force_all:
            force_candidates = True
            force_predictions = True
            force_output = True
        
        candidates = self.get_candidates(htid, save=save_candidates, force=force_candidates, **ann_args)
        predictions = self.get_model_predictions(htid, candidates, save=save_predictions, force=force_predictions)
        if skip_json_output:
            print('Skipping json output')
            return predictions
        else:
            data_entry = self.export_structured_data(htid, predictions, save=save_output, force=force_output)
            return data_entry
    
    def _get_simmats_from_candidates(self, candidates, max_size=150, reshape=False, include_wem=True):
        '''
        For a target HTID, find match candidates with ANN, then return the target-candidate similarity matrices. 
        Sim mats are unrolled, unless reshape argument is given. 
        
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
        
    def get_model_predictions(self, htid, candidates, model_path=None, metadb_path=None, save=False, force=False):
        '''
        Take left-right candidates and run it through the tensorflow model.
        '''
        outpath = os.path.join(self.data_dir, utils.id_to_stubbytree(htid, format='predictions.parquet'))
        
        if not force and os.path.exists(outpath):
            print('Predictions already found: {}'.format(outpath))
            predictions = pd.read_parquet(outpath)
            return predictions
        
        rightindex, inputs = self._get_simmats_from_candidates(candidates, reshape=(150,150,1))
        predictions = self._predict_from_simmat(rightindex, inputs, model_path, metadb_path)
           
        if save:
            os.makedirs(os.path.split(outpath)[0], exist_ok=True) # Create directories if needed
            predictions.to_parquet(outpath, compression='snappy')
        
        return predictions
    
    def titleann(self, path=None, dims=50, prefault=False, force=False):
        if self._titleann and not force:
            return self._titleann
        else:
            if not path:
                path = self.config['title_ann_path']
            self._titleann = TitleAnnoy(path, dims)
            self._titleann.load()
        
    def get_meta_candidates(self, htid, sim_titles=True, same_authors=True, max_dist=.35):
        assert sim_titles or same_authors
        titleann = self.titleann()

        if sim_titles:
            idnum = titleann.htid2id[htid]

            results = titleann.u.get_nns_by_item(idnum, n=25, include_distances=True)
            if (results[1][-1] < .3):
                results = titleann.u.get_nns_by_item(idnum, n=100, include_distances=True)

            results = dict(zip(*results))

            result_htids = [titleann.id2htid[id] for id in results.keys()]

            if htid not in result_htids:
                result_htids.append(htid)

            # add self
            meta = dd.read_parquet(self.config['metadb_path'], engine='pyarrow-dataset',
                                               columns=['title', 'author'],
                                               filters=[('htid', 'in', tuple(result_htids))]).compute()
            meta = meta.loc[result_htids]
            meta['distance'] = results.values()
            # Trim to just 'pretty similar'
            meta = meta[meta.distance <= max_dist]
        else:
            meta = dd.read_parquet(self.config['metadb_path'], engine='pyarrow-dataset',
                                               columns=['title', 'author'],
                                               filters=[('htid', '==', htid)]).compute()

        author = meta.loc[htid, 'author']

        if same_authors:
            same_aut = dd.read_parquet(self.config['metadb_path'], engine='pyarrow-dataset',
                                       columns=['title', 'author'],
                                       filters=[('author', '==', author)]).compute()
            return meta, same_aut
        else:
            return meta
    
    def _predict_from_simmat(self, rightindex, inputs, model_path=None, metadb_path=None):
        '''
        Take Similarity matrix inputs to tensorflow model and format output.
        '''
        relatedness_weights = {'SWSM': 1, 'SWDE': 0.7, 'WP_DV': 0.7, 'PARTOF': 0.6,
                               'CONTAINS': 0.6, 'OVERLAPS': 0.6, 'AUTHOR': 0.3,
                               'SIMDIFF': 0.1, 'GRSIM': 0.4, 'RANDDIFF': 0}
        
        model = self.tf_model(model_path)
        weights = [relatedness_weights[x] for x in judgment_labels]

        predictions = model.predict(inputs)
        predictions = pd.DataFrame(predictions, columns=judgment_labels)
        predictions['htid'] = rightindex
        best_predictions = np.argmax(predictions[judgment_labels].values, axis=1)
        predictions['guess'] = [judgment_labels[i] for i in best_predictions]
        predictions = predictions.sort_values(judgment_labels, ascending=False)
        # pyarrow-dataset is awesome fast
        filters = ('htid', 'in', tuple(predictions.htid.tolist()))
        if not metadb_path and 'metadb_path' not in self.config:
            raise Exception('Need reference to HathiMeta path (or just a parquet version of the Hathifiles)'
                           'in config file (metadb_path)')
        metadf = dd.read_parquet(self.config['metadb_path'], engine='pyarrow-dataset',
                                 columns=['title','description','author', 'rights_date_used', 'oclc_num', 'isbn'],
                                 filters=[filters]).compute()
        predictions = predictions.merge(metadf.reset_index(), on='htid')
        predictions['relatedness'] = np.average(predictions[judgment_labels], weights= weights, axis=1) # Weighted average of probabilities, with emphasis on SWSM
        
        return predictions
    
    def export_structured_data(self, htid, predictions, target=None, save=False, force=False):
        '''
        target: Series of metadata for target - this is *loaded* in this method, so only supply if you already
            have it in memory and don't want to do the lookup again.
        '''
        outpath = os.path.join(self.data_dir, utils.id_to_stubbytree(htid, format='saddl.json'))
        if not force and os.path.exists(outpath):
            print('Dataset already found: {}'.format(outpath))
            try:
                with open(outpath, mode='r') as f:
                    data_entry = json.load(f)
                    return data_entry
            except json.JSONDecodeError:
                print("loading error. Will ignore loading")

        if not target:
            target = dd.read_parquet(self.config['metadb_path'], engine='pyarrow-dataset',
                                     filters=[('htid', '==', htid)]
                                    ).reset_index().compute().iloc[0]
        
        base_meta = ['htid', 'title', 'author', 'description', 'rights_date_used', 'oclc_num', 'isbn']
        data_entry = dict(volume=target[base_meta].to_dict())
        data_entry['volume']['link'] = "http://hdl.handle.net/2027/" + target['htid']
        data_entry['related_metadata'] = dict()
        data_entry['relationships'] = dict()
        data_entry['recommendations'] = dict()

        aut_prints = predictions.author.apply(alpha_fingerprint)
        target_print = alpha_fingerprint(target.author)
        by_author = predictions[aut_prints == target_print]

        # Add Collected Metadata
        def unique_nontarget_values(field, limit=['SWSM', 'SWSE'], df=by_author):
            diff = (df[field] != target[field]) if target[field] else True
            uniq = df[df.guess.isin(limit) & diff][field].unique().tolist()
            return uniq if len(uniq) else []
        data_entry['related_metadata']['other years'] = unique_nontarget_values('rights_date_used')
        data_entry['related_metadata']['other titles'] = unique_nontarget_values('title')
        data_entry['related_metadata']['other OCLC numbers'] = unique_nontarget_values('oclc_num')
        data_entry['related_metadata']['other enumchron values'] = unique_nontarget_values('description')
        data_entry['related_metadata']['titles within this work'] = unique_nontarget_values('title', ["CONTAINS"])
        data_entry['related_metadata']['titles of works that contain this work'] = unique_nontarget_values('title', ["PARTOF"])

        # Add Same Work Info
        def get_dict_by_guess(guess):
            a = by_author[by_author.guess == guess].sort_values(guess, ascending=False)
            if a.empty:
                return []
            a = a[base_meta + [guess]]
            a = a.rename(columns={'rights_date_used': 'year', guess: "confidence"})
            a['confidence'] = a['confidence'].multiply(100).astype(int)
            return a.to_dict(orient='records')

        data_entry['relationships']['identical works'] = get_dict_by_guess("SWSM")
        data_entry['relationships']['different expressions'] = get_dict_by_guess("SWDE")
        data_entry['relationships']['other volumes of the larger work'] = get_dict_by_guess("WP_DV")
        data_entry['relationships']['this work contains'] = get_dict_by_guess("CONTAINS")
        data_entry['relationships']['this work is a part of'] = get_dict_by_guess("PARTOF")

        other_works = predictions[~predictions.guess.isin(['SWSM', 'SWDE', 'WP_DV', 'CONTAINS', 'PARTOF'])]
        recs = other_works[other_works.relatedness > 0.05].sort_values('relatedness').head(20)
        data_entry['recommendations']['related authors'] = unique_nontarget_values('author', judgment_labels, df=recs)
        data_entry['recommendations']['similar books'] = recs[base_meta].rename(columns={'rights_date_used': 'year'}).to_dict(orient='records')

        if save:
            os.makedirs(os.path.split(outpath)[0], exist_ok=True) # Create directories if needed
            with open(outpath, mode='w') as f:
                json.dump(data_entry, f)
            
        return data_entry

def alpha_fingerprint(s):
    ''' Fingerprint that is just alphabetical characters, lowercased and sorted.'''
    if type(s) is not str:
        return s
    else:
        return "".join(sorted([b for b in s.lower() if b.isalpha()]))
    
def main():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument("--data-root", type=str, default='/data/saddl/full/',
                        help="Location to save stubbytree data file outputs")
    ann_parser = subparsers.add_parser("Candidates",
                                       help="Save candidate relationships from ANN")
    prediction_parser = subparsers.add_parser("Predictions",
                                       help="Run candidates through SaDDL model to get predicted relationship.")

    # Configure for the MTAnnoy candidate retrieval
    for subparser in [ann_parser, prediction_parser]:
        subparser.add_argument('--ann-path', type=str, default=None,
                                help="Location of MTAnnoy index. Default is None, which tries to fall " \
                                "back on what's in the config file")
        subparser.add_argument('--ann-dims', type=int, default=50,
                               help='Number of dimensions for the MTAnnoy index.')
        subparser.add_argument("--results-per-chunk", "-n", type=int, default=300,
                               help="Number of ANN results to return per chunk")
        subparser.add_argument("--min-count", type=int, default=2,
                               help="Min number of matching chunks between books.")
        subparser.add_argument("--max-dist", type=float, default=.25,
                               help="Maximum distance between matching chunks.")
        subparser.add_argument("--htid-in", type=argparse.FileType('r'), default=None,
                               help='File of HTIDs to process. If set, htids args provided on the command line are ignored.')
        subparser.add_argument("htids", nargs='*', help='HTIDs to process. Alternately, provide --htid-in')
        subparser.add_argument('--prefault', action='store_true',
                               help='Load ANN into memory.')
        subparser.add_argument("--search-k", type=int, default=-1,
                               help="ANN search k parameter.")
    
    ann_parser.add_argument("--overwrite", action="store_true",
                            help="Overwrite files if they already exist. Otherwise, they're skipped")
    
    # Args for prediction parser
    prediction_parser.add_argument('--model-path', type=str, default=None,
                            help="Location of SaDDL model. Default is None, which tries to fall " \
                            "back on what's in the config file")
    prediction_parser.add_argument("--force-candidates", action="store_true",
                            help="Reprocess and overwrite candidate raising process if they already exist")
    prediction_parser.add_argument("--force-predictions", action="store_true",
                            help="Reprocess and overwrite model inference if it's already been saved")
    prediction_parser.add_argument("--force-json", action="store_true",
                            help="Reprocess and overwrite final JSON files formatting if it's already been done")
    prediction_parser.add_argument("--skip-json-output", action="store_true",
                            help="Just do model inference and save the raw data in parquet, without formatting for " \
                                   "dataset output.")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    saddlr = Saddler(data_dir=args.data_root)
    
    if args.command == 'Candidates':
        # Pre-load MTAnnoy. Unnecessary, but more readable below
        saddlr.mtannoy(ann_dims=args.ann_dims, ann_path=args.ann_path, prefault=args.prefault)
        
        if args.htid_in:
            htids = [htid.strip() for htid in args.htid_in]
        else:
            htids = args.htids
        
        starttime = time.time()
        skipped = 0
        for i, htid in enumerate(htids):
            try:
                outpath = os.path.join(args.data_root, utils.id_to_stubbytree(htid, format='ann.parquet'))
                if not args.overwrite and os.path.exists(outpath):
                    print('File already found: {}'.format(outpath))
                    skipped += 1
                    continue
    
                results = saddlr.get_candidates(htid,
                                                n=args.results_per_chunk, 
                                                min_count=args.min_count, 
                                                max_dist=args.max_dist,
                                                search_k=args.search_k,
                                                force=args.overwrite,
                                                save=True)
                
                if i % 2 == 1:
                    progress = (time.time() - starttime)/60
                    remaining = progress/(i-skipped+1) * (len(htids)-i-skipped)
                    print(f"{i-skipped+1}/{len(htids)-skipped} completed in {progress:.1f}min (Est left: {remaining:.1f}min)")
            except KeyboardInterrupt:
                raise
            
            except:
                print("Issue with {}".format(htid))
                
    elif args.command == "Predictions":
        # Pre-load TF Model, for readability
        saddlr.tf_model(args.model_path)
        
        if args.htid_in:
            htids = [htid.strip() for htid in args.htid_in]
        else:
            htids = args.htids
    
        for i, htid in enumerate(htids):
            try:
                saddlr.get_predictions(htid, save_all=True,
                                       force_candidates=args.force_candidates,
                                       force_predictions=args.force_predictions,
                                       force_output=args.force_json,
                                       skip_json_output=args.skip_json_output,
                                       ann_args=dict(n=args.results_per_chunk, min_count=args.min_count,
                                                     max_dist=args.max_dist, search_k=args.search_k,
                                                     ann_path=args.ann_path, ann_dims=args.ann_dims,
                                                     prefault=args.prefault)
                                      )
                
            except KeyboardInterrupt:
                raise
            
            except:
                raise
                print("Issue with {}".format(htid))
                
        
        

if __name__ == '__main__':
    main()
