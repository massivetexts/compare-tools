'''
Helper functions for clustering volume<->volume predictions into manifestation and group works.

These are all very context-specific right now - reading scripts/VolumesToWorks.ipynb for context on how they are used.
'''
from sknetwork.clustering import Louvain
from sknetwork.data import convert_edge_list
import pandas as pd
from compare_tools.train_utils import judgment_labels
import pyarrow.parquet as pq

def high_conf_predictions(con, from_table="clean_predictions", field='swsm', min_probability=.99):
    ''' Return all the predictions rows with a high confidence in the given field.
    Only returns where target < candidate, so flipped duplicates are not included.
    
    Field can be a list of multiple fields, in which case the *sum* 
    of the softmax probabilities is used'''
    if from_table.endswith('.parquet'):
        from_table = f"'{from_table}'"
    if type(field) is list:
        field_select = ", ".join(field)
        field_value = '(' + "+".join(field) + ')'
    else:
        field_select= field
        field_value = field
    return con.execute(f'SELECT target, candidate, {field_select} FROM {from_table} WHERE {field_value} > {min_probability} AND target < candidate').fetch_df()


def network_cluster(inferences, weighted=False, weight_col='swsm'):
    ''' Do a simple Louvain clustering on the nodes. This is unweighted, so you want
    high-confidence edges only: the main value is how fast it is.
    '''
    cols = ['target', 'candidate']
    if weighted:
        cols.append(weight_col)
    edge_list = list(inferences[cols].itertuples(index=False))
    graph = convert_edge_list(edge_list)
    
    louvain = Louvain(modularity='newman', tol_aggregation=.001, shuffle_nodes=True)
    labels = louvain.fit_transform(graph.adjacency)
    cluster_labels = pd.DataFrame(zip(labels, graph.names), columns=['label', 'htid'])
    return cluster_labels

def label_single_clusters(con, count_from=1, 
                          from_table='clean_predictions', already_labeled=[]):
    '''Add labels to single htids that didn't have a matching high-conf SWSM judgment'''
    if from_table.endswith('.parquet'):
        from_table = f"'{from_table}'"
    unique_targets = con.execute(f"SELECT DISTINCT target FROM {from_table}").fetch_df()
    singletons = list(set(unique_targets.target).difference(already_labeled))
    extra_labels = range(count_from, count_from+len(singletons))
    return pd.DataFrame(zip(extra_labels, singletons), columns=['label', 'htid'])

def cluster_all(con, inferences, id_col='man_id', from_table='clean_predictions', save_to=False):
    ''' Cluster the network relationships provided, *and* give cluster ids to the 
    remaining volumes that didn't have edges in this network.
    
    con: DuckDB connection
    inferences: The edges of relationship inferences
    id_col: what to name the class label. I.e. 'man_id' or 'work_id'
    '''
    cluster_labels = network_cluster(inferences)
    # Add labels to single htids that didn't have a matching high-conf SWSM judgment
    extra_labels = label_single_clusters(con, already_labeled=cluster_labels.htid,
                                         count_from=cluster_labels.label.max()+1,
                                        from_table=from_table)
    
    cluster_labels = pd.concat([cluster_labels, extra_labels])
    cluster_labels = cluster_labels.rename(columns={'label':id_col})
                                                    
    if cluster_labels.htid.dtype == 'float64':
            cluster_labels.htid = cluster_labels.htid.astype(int)

    if save_to:
        # Save to disk rather than a new table - since duckdb is fairly new,
        # I prefer keeping it read-only until necessary
        cluster_labels.to_parquet(save_to)
    return cluster_labels

def join_clustered_predictions(con, label_fname, out_fname, id_col='label'):
    ''' This query does two things: 
        1) remaps the target/candidate htids to their labels, then 
        2) does a weighted average of probability and sum of row counts (i.e. aggregating down to a single target label/candidate label row)
    '''
    
    select_cols = ", ".join(['"{0}"'.format(x.lower()) for x in judgment_labels]+['relatedness', '"count"'])
    weighted_avg_cols = ", ".join(['SUM("{0}" * "count") / SUM("count") as "{0}"'.format(x.lower()) for x in judgment_labels])
    df = con.execute('''
    COPY
        (SELECT target, candidate, ''' + weighted_avg_cols + ''', CAST(sum("count") AS INTEGER) as "count"
        FROM (
            SELECT clusters.''' + id_col + ''' as target,
                   clusters2.''' + id_col + ''' as candidate,
                   ''' + select_cols + '''
            FROM clean_predictions
            JOIN \'''' + label_fname + '''\' as clusters on clusters.htid == clean_predictions.target
            JOIN \'''' + label_fname + '''\' as clusters2 on clusters2.htid == clean_predictions.candidate
        ) ref
        GROUP BY target, candidate)
    TO \'''' + out_fname + '''\' (FORMAT 'parquet')
    ''').fetch_df()
    return df


def merge_labels(con, oldtable, newtable, labelname, last_labelname=None,
                debug=False):
    ''' When working with intermediate labels (i.e. you clustered htids, then 
    clustered those clusters), take the old cluster label reference and reassign htids to new 
    labels.'''
    
    if not last_labelname:
        last_labelname = labelname

    # get column names from schema, to preserve old labels
    pfile = pq.read_table(oldtable)
    pass_columns = [x for x in pfile.column_names if '_pass' in x]
    select_cols = [f'new_label.{labelname}'] + ["old_label."+x for x in pfile.column_names if x.endswith('id') and x != labelname]
    if labelname != last_labelname:
        select_cols.append(f"old_label.{last_labelname}")
    select_cols = select_cols + pass_columns
    if labelname in pfile.column_names:
        # rename last head id column to an intermediate 'pass' column
        pass_nums = [int(x.split('_')[-1].replace('pass', '')) for x in pass_columns]
        max_pass = max(pass_nums) if len(pass_nums) else 0
        lastpass_name = f'{labelname}_pass{max_pass+1}'
        select_cols += [f'old_label.{labelname} as {lastpass_name}']
    
    q = '''
    SELECT ''' + ",".join(select_cols) + '''
    FROM \'''' + oldtable + '''\' old_label
    JOIN \'''' + newtable + '''\' AS new_label ON old_label.''' + last_labelname + ''' == new_label.htid
    '''
    if debug:
        print(q)
    df = con.execute(q).fetch_df()
    return df


def process_label_stats(con, label_fname, out_fname=None, out_table=None, id_col='man_id', print_query=False):
    ''' Get stats on serial and gov doc prevalence in a cluster '''
    assert out_fname or out_table
    assert not (out_fname and out_table)
    main_statement = '''
        SELECT *, gov_count/CAST(label_count AS float) as gov_prop, 
            serial_count/CAST(label_count AS float) as serial_prop
        FROM (
            SELECT ''' + id_col + ''', 
                    COUNT(*) as label_count,
                    COUNT(*) filter (where us_gov_doc_flag == True) as gov_count, 
                    COUNT(*) filter (where bib_fmt == 'SE') as serial_count
            FROM "''' + label_fname + '''" as clusters 
            JOIN meta on clusters.htid == meta.htid 
            GROUP BY ''' + id_col + '''
            ) label_stats
    '''
    if out_fname:
        q = f"COPY ({main_statement}) TO '{out_fname}' (FORMAT 'parquet')"
    elif out_table:
        q = f"CREATE TABLE {out_table} AS ({main_statement})"
    if print_query:
        print(q)
    con.execute(q)