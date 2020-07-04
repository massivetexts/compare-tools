import pandas as pd

class HathiMeta():
    
    def __init__(self, db_path=None, default_fields='*'):
        '''
        An SQLite-backed metadata accessor for HathiTrust information.
        
        db_path: Location of sql database. The default is an in-memory database.
        The table names are not customizable. If creating a new database, data will
        need to be imported with create_db(meta_path)
        '''
        
        from sqlalchemy import create_engine
        if not db_path:
            conn_string = 'sqlite://'
        else:
            conn_string = 'sqlite:///' + db_path
        
        self.engine = create_engine(conn_string, echo=False)
        
        self.field_list = None
        self.default_fields = default_fields
        
    def create_db(self, meta_path):
        ''' Import Hathifiles CSV to the DB, rewriting if necessary.
        Eventually, logic for extending the data beyond the Hathifiles and better
        error catching for CSV-problems should exist here.
        '''
        
        # Needs smarter logic for addressing bad CSV formatting
        chunks = pd.read_csv(meta_path, chunksize=25000, index_col='htid')

        with self.engine.connect() as conn:
            for i, chunk in enumerate(chunks):
                print(i, end=', ')
                if i < 1:
                    chunk.to_sql('meta', conn, if_exists='replace')
                    self.field_list = chunk.columns.values
                else:
                    chunk.to_sql('meta', conn, if_exists='append')
        
            res = conn.execute('CREATE INDEX meta_htid ON meta (htid);')
    
    def extend_db(self, df):
        ''' Add data to the metadata by passing a dataframe with htid and the 
        new columns.'''
        newcols = [c for c in df.columns.tolist() if c !='htid']
        df.to_sql('tmp', self.engine, if_exists='replace', index=False)

        with self.engine.connect() as conn:
            # Left join to preserve all original rows
            conn.execute(('CREATE TABLE tmp2 AS SELECT meta.*,{} FROM meta '
                         'LEFT JOIN tmp ON meta.htid=tmp.htid').format(
                             self._field_call(newcols))
                        )
            conn.execute('DROP TABLE meta')
            conn.execute('DROP TABLE tmp')
            conn.execute('ALTER TABLE tmp2 RENAME TO meta')

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
        sql = single_item_template.format(self._field_call(fields), htid)
        return pd.read_sql_query(sql, self.engine).iloc[0]
    
    def get_where(self, where_clause, fields=None):
        '''
        Retrieve all records WHERE {clause}
        '''
        template = 'SELECT {} FROM meta WHERE {}'
        if not fields:
            fields = self.default_fields
        sql = template.format(self._field_call(fields), where_clause)
        return pd.read_sql_query(sql, self.engine)
    
    def sample(self, n=1, fields=None):
        ''' Return a single random volume. '''
        random_item_template = "SELECT {} FROM meta ORDER BY RANDOM() LIMIT {};"
        if not fields:
            fields = self.default_fields
        sql = random_item_template.format(self._field_call(fields), n)
        results = pd.read_sql_query(sql, self.engine)
        if n == 1:
            return results.iloc[0]
        return results
    
    def get_fields(self, fields=None, chunksize=None, offset=None, limit=None):
        '''Retrieve full table, filtered to the fields specified or '*'. Can be chunked.
        '''
        if not fields:
            fields = self.default_fields
        sql = 'SELECT {} FROM meta'.format(self._field_call(fields))
        return pd.read_sql_query(sql, self.engine, chunksize=chunksize)
    
    def _field_call(self, q):
        if q == '*':
            return q
        else:
            assert type(q) is list
            return ",".join(q)
    
    def __len__(self):
        with self.engine.connect() as conn:
            return conn.execute('SELECT COUNT(*) FROM meta').fetchone()[0]
    
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
    title = title.split('; by')[0]
    title = re.sub('[\.,;\]\)\[] +?[\(\[]?(assembl|photo|arrang|select|compil|record|collect|edit|translat).{0,100}by.*\.?', '', title, flags=re.IGNORECASE)
    title = re.sub('[\.\,] [bB]y (the .{0,30})?([A-Z]\w+ [A-Z\w+]|author).*\.?', '', title)
    return title