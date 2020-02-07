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
    
    def random(self, fields=None):
        ''' Return a single random volume. '''
        random_item_template = "SELECT {} FROM meta ORDER BY RANDOM() LIMIT 1;"
        if not fields:
            fields = self.default_fields
        sql = random_item_template.format(self._field_call(fields))
        return pd.read_sql_query(sql, self.engine).iloc[0]
    
    def get_fields(self, fields=None, chunksize=None):
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

def get_json_meta(htid, parquet_root):
    ''' Quickly read a pairtree-organized metadata file that accompanies 
    the Parquet Feature Reader export.'''
    from htrc_features import utils
    import ujson as json
    path = parquet_root + utils.id_to_rsync(htid).replace('json.bz2', 'meta.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def meta_compare(left_htid, right_htid, hathimeta, fields=['title', 'author', 'oclc_num', 'page_count', 'description']):
    ''' Compare basic metadata on two books'''
    a = hathimeta[left_htid][fields]
    b = hathimeta[right_htid][fields]
    return pd.DataFrame([a,b], index=['left', 'right'])