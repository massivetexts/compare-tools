import numpy as np
import tensorflow as tf

judgment_labels = ['SWSM', 'SWDE', 'WP_DV', 'PARTOF', 'CONTAINS', 'AUTHOR', 'DIFF']
# Create a {code: index} dict
judgment_label_ref = dict(zip(judgment_labels, range(len(judgment_labels))))

def df_to_tfrecords(X, output_file, label_ref='default'):
    '''
    Take a dataframe where
        - the first X*X columns are features for a pairwise comparison between
    two books, up to X chunks, with extra chunks padded with zeros. X doesn't need
    to be known here - the assumption is that everything but the last four columns
    is a feature.
        - the final four columns have 'left', 'right', 'judgment', and 'notes'
        
    This is saved to a TFRecord file, a Proto-based format that works well with Tensorflow.
    
    Judgment labels are converted to an integer and then one-hot encoded. To safely keep a consistent
    label->label_int key, they are hardcoded. Change only for your own use cases. If adding
    classes for the present problem, add to the end of the current list. Just in case, the
    string judgment is still kept in the TFRecord file.
    '''
    
    if label_ref == 'default':
        label_ref = judgment_labels

    writer = tf.io.TFRecordWriter(output_file)
    for i, row in X.iterrows():
        serialized = _serialize_series(row, label_ref)
        writer.write(serialized)
    writer.close()
    
def _serialize_series(row, label_ref):
    # Convert judgment to one-hot encode
    if 'judgment' in row and row['judgment'] in label_ref:
        y = tf.one_hot(label_ref[row['judgment']], len(label_ref))
    else:
        y = tf.one_hot(-1, len(labels))
            
    feature = {
          'X': tf.train.Feature(float_list=tf.train.FloatList(value=row.iloc[:-4].values.flatten())),
          'left': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['left'].encode('utf-8')])),
          'right': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['right'].encode('utf-8')])),
          'judgment': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['judgment'].encode('utf-8')])),
          'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y)),
          'notes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['notes'].encode('utf-8')])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    return serialized

def parse_comparison_records(example_proto, labels='default', input_shape=(50,50,1), parse_single=False):
    '''Definition for reading TFRecords. Input shape should be three-dimensional, with a channel at the end.'''
    if labels == 'default':
        labels = judgment_labels
    features = {
        'X': tf.io.FixedLenFeature(input_shape, tf.float32),
        'left': tf.io.FixedLenFeature([], tf.string),
        'right': tf.io.FixedLenFeature([], tf.string),
        'judgment': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([len(labels)], tf.int64),
        'notes': tf.io.FixedLenFeature([], tf.string)
    }
    '''  parse_example gives a "Shape must be rank 1" error on some versions of tensorflow. You can change
    to parse_single_example, but it's more transparent to just batch the data first , even if
    it is a batch of 1; e.g. ds = ds.batch(1)
    
    '''
    if parse_single:
        parsed_features = tf.io.parse_single_example(example_proto, features)
    else:
        parsed_features = tf.io.parse_example(example_proto, features)
    return parsed_features

def processStats(stats, remove_cols=['LSize', 'RSize', 'minSize']):
    '''Add derivative columns to stats'''
    stats = stats.copy()
    replaced_cols = remove_cols
    sw_sizes = [int(col[2:-3])/100 for col in stats.columns if col.startswith('SW')]
    for t in sw_sizes:
        for s in ['L', 'R']:
            stats[s+'SW{:04.0f}Prop'.format(t*100)] = stats['SW{:04.0f}Len'.format(t*100)] / stats[s+'Size']
        replaced_cols.append('SW{:04.0f}Len'.format(t*100))
            
    # Log Transforms, robust scale, no +1
    to_transform_cols = ['gloveMeanSim', 'gloveLMeanMinSim', 'gloveRMeanMinSim', 'gloveLTruncSim', 'gloveRTruncSim']
    to_transform_cols += [col for col in stats.columns if 'Quantile' in col]
    for col in to_transform_cols:
        #To update: Newer versions of daskml have a robust scaler
        #stats[col+'Transform'] = preprocessing.robust_scale(stats[col].apply(np.log))
        stats[col+'Transform'] = stats[col].add(.1**7).apply(np.log)
    replaced_cols += to_transform_cols
    
    stats['propSize'] = stats['LSize'] / stats['RSize']
    
    # Return a dataframe with the new important stats cols, and another with the ones that have been superceded
    a = stats.loc[:,[col for col in stats.columns if col not in replaced_cols]]
    a = stats.loc[:, ['left', 'right'] + [col for col in a.columns if col not in ['left', 'right']]]
    b = stats.loc[:,list(set(replaced_cols))]
    return (a, b)

def print_most_important_for_forest(forest, cols):
    print('most important')
    # from https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, cols[indices[f]], importances[indices[f]]))