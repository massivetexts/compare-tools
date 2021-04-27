import pandas as pd
import dask.dataframe as dd
import numpy as np
import tempfile
import os
from dask.diagnostics import ProgressBar

class HathiMeta():
    HTTYPES = dict(htid=str, access=str, rights=str, ht_bib_key=str, description=str, source=str,
               source_bib_num=str, oclc_num=str, isbn=str, issn=str, lccn=str, title=str,
               imprint=str, rights_read_code=str, rights_timestamp=str, us_gov_doc_flag=int,
              rights_date_used=float, pub_place=str, lang=str, bib_fmt=str, collection_code=str,
              content_provider_code=str, responsible_entity_code=str, digitization_agent_code=str,
              access_profile_code=str, author=str)
    
    def __init__(self, data_path=None, default_fields='*'):
        '''
        A Dask+Parquet accessor for Hathifiles info. Replaces a former SQLite-backed 
        class, which would inexplicably need rebuilding from time to time.
        
        db_path: Location of parquet files.
        '''
        
        self.data_path = data_path
        self.field_list = list(self.HTTYPES.keys())
        self.default_fields = None
        try:
            self.ddf = dd.read_parquet(data_path, compression='snappy')
        except FileNotFoundError:
            self.ddf = None
            print('No dataset exists yet. Run create_db to parse a raw CSV file.')
            
    def create_db(self, meta_path, chunk_size=100000):
        ''' Process Hathifiles CSV for Dask-appropriate parquet.'''
        
        import os
        import glob
        fdir, fname = os.path.split(meta_path)

        chunks = pd.read_csv(meta_path, dtype=self.HTTYPES, chunksize=chunk_size)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmppaths = []
            for i, chunk in enumerate(chunks):
                print(i, end=',')
                tmppath = os.path.join(tmpdirname, os.path.splitext(fname)[0] + f'.{i}.parquet')

                chunk.to_parquet(path=tmppath, engine='pyarrow', compression='snappy')
                tmppaths.append(tmppath)
            print('Done writing temp files; indexing and saving final')

            # Have dask read these unindexed parquet files, set the index, and save the final partitioned version
            dd.read_parquet(tmppaths).set_index('htid').to_parquet(self.data_path, compression='snappy')
            
        self.ddf = dd.read_parquet(self.data_path, compression='snappy')

    def extend_meta(self, df):
        ''' Add data to the metadata by passing a dataframe with htid and the 
        new columns.'''
        with ProgressBar():
            new_ddf = self.ddf.join(df, on='htid')
            new_ddf.to_parquet(self.data_path+'.new')
        
        print('Extended files created. Deleting old files')
        for file in os.listdir(self.data_path):
            fname = os.path.join(self.data_path, file)
            os.remove(fname)
        os.removedirs(self.data_path)
        os.rename(self.data_path+'.new', self.data_path)
        
        self.ddf = dd.read_parquet(self.data_path, compression='snappy')
        
    def extend_db(self, df):
        self.extend_meta(df)

    def full_table(self):
        print('Deprecated: `full_table` is just a wrapper for HathiMeta.get_fields.')
        return self.get_fields()
        
    def get_volume(self, htid, fields=None):
        '''Retrieve metadata about a Volume by it's HTID number.
        
        fields = a list of columns to return, or '*'. If none, uses the default
            set on init.
        '''
        single_item_template = 'SELECT {} FROM meta WHERE htid = "{}"'
        if not fields:
            fields = self.default_fields
        # For consistency with prior version, we reset the index to keep 'htid' as a column
        return self.ddf.loc[htid, fields].compute().reset_index().iloc[0]
    
    def get_where(self, where_clause, fields=None):
        '''
        Retrieve all records WHERE {clause}. Using query syntax from pandas.
        '''
        if not fields:
            fields = self.default_fields
        return self.ddf.query(where_clause).loc[slice(None), fields].compute()
    
    def sample(self, n=1, fields=None):
        ''' Return a single random volume for a random partition '''
        randpart = np.random.randint(self.ddf.npartitions)
        if not fields:
            fields = self.default_fields
        results = self.ddf.get_partition(randpart).loc[slice(None), fields].compute().sample(n)
        results = results.reset_index()
        if n == 1:
            return results.iloc[0]
        return results
    
    def get_fields(self, fields=None, chunksize=None, by_partition=False, offset=None, limit=None):
        '''Retrieve full table, filtered to the fields specified or '*'. Chunks no longer work.
        '''
        if not fields:
            fields = self.default_fields
            
        df = self.ddf.loc[slice(None), fields]
        
        if chunksize:
            raise Exception('Chunking removed with parquet backend. Try by_partition=True instead')
        if offset:
            df = df.iloc[offset:]
        if limit:
            df = df.iloc[:limit]
        if by_partition:
            return self._partition_yielder(df)
        else:
            return df.compute()
    
    def _partition_yielder(self, df):
        ''' Return results as a generator, by partition'''
        for i in range(df.npartitions):
            yield df.get_partition(i).compute()
        
    def _field_call(self, q):
        if q == '*':
            return None
        else:
            assert type(q) is list
            return q
    
    def __len__(self):
        return len(ddf)
    
    def __getitem__(self, label):
        return self.get_volume(label, fields=None)


def get_json_meta(htid, parquet_root, id_resolver='pairtree'):
    ''' Quickly read a pairtree-organized metadata file that accompanies 
    the Parquet Feature Reader export.'''
    from htrc_features import utils as efutils
    import ujson as json
    if id_resolver == 'pairtree':
        path = parquet_root + efutils.id_to_rsync(htid).replace('json.bz2', 'meta.json')
    elif id_resolver =='stubbytree':
        from compare_tools.utils import StubbytreeResolver
        path = parquet_root + StubbytreeResolver.id_to_stubbytree(None, htid, format=None) + '.meta.json'
    else:
        raise Exception('Unexpected id_resolver argument')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def meta_compare(left_htid, right_htid, hathimeta, fields=['title', 'author', 'oclc_num', 'page_count', 'description']):
    ''' Compare basic metadata on two books'''
    a = hathimeta[left_htid][fields]
    b = hathimeta[right_htid][fields]
    return pd.DataFrame([a,b], index=['left', 'right'])

def clean_description(s):
    ''' Takes a column of enumchron/description fields and normalizes them.
    
    e.g. vol. 1 -> v1
        v.001 -> v.1
        cop.1 -> c.1
        Pt1 -> pt.1
        
    '''
    new = (s.str.lower().str.replace('v. ', 'v.')
            .str.replace('^(\d+)$', r'v.\1')
            .str.replace('copy ?|cop.', 'c.')
            .str.replace('\.[0 ]+', '.')
            .str.replace('\((.*?)\)', r'\1')
            .str.replace('vol.', 'v.')
            .str.replace('\.$', '')
            .str.replace('([v|pt|c])(\d)', r'\1.\2')
    )
    return new

def clean_title(title):
    ''' Clean a title field'''
    import re
    # Remove up to two trailing '/'-separated sections
    title = title[::-1].split('/', 2)[-1][::-1].strip()
    title = title.split('; by')[0].split('[by]')[0].split('; illustrated by')[0]
    title = re.sub('[\.,;\]\)\[] +?[\(\[]?(assembl|photo|arrang|select|compil|record|collect|edit|translat).{0,100}by.*\.?', '', title, flags=re.IGNORECASE)
    title = re.sub('[\.\,] [bB]y (the .{0,30})?([A-Z]\w+ [A-Z\w+]|author).*\.?', '', title)
    return title.strip().strip('.')