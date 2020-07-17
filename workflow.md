This is the (in-progress) document outlining the SaDDL workflow.


## Pipeline Overview

```
EF Files
    ↳ Vector files
       ↳ MTAnnoy Index
           ↳ Comparison Candidates
                ↙
       ↳ Compare_tools (a)
                ↖
Hathifiles → HathiMeta 

Hand-coded relationships    Metadata inferred relationships
                ↳ Ground truth ↲
                       ↳ Train / Test
                                   ↘
Compare_tools → Feature Extraction → Classifier
```

## Optional - Converting Extracted Features (EF) files to Parquet

The original Extracted Features Dataset is in bzip-encoded JSON files. These are costly to decompress and parse. To avoid this bottleneck, we've created an alternative format, which stores [Parquet](https://parquet.apache.org/)-based alternatives. This functionality is in the newest [Massive Texts fork of the HTRC Feature Reader](https://github.com/massivetexts/htrc-feature-reader/). Of course, converting to Parquet requires the cycles to open the JSON.BZ2 files at least once, but if you intend to work with the files often, or if you want to offload some of the processing time to a preparatory stage, it is a sensible option.

A parallizable script for this conversion is in [scripts/convert-to-parquet.py](scripts/convert-to-parquet.py).

A second optional preparatory step is flattening the extremely deep, slow Pairtree format of the Extracted Features Dataset. One option that we use is a custom Stubbytree structure, which organized HTIDs into directories based on every third character of the volume id (e.g. where an htid is `libid.volid`, its EF file will be in `libid\volid[::3]\libid.volid.json.bz2`). A script for converting Pairtree to Stubbytree and gradually remove the Pairtree directories is in [scripts/pairtree_to_stubbytree.py](scripts/pairtree_to_stubbytree.py).

## HTID class - Convenient access to different data sources

To allow for easier juggling of the various data sources (metadata, vectorfiles, EF), the HTID class allows lazy-loading of the information that you need when you need it. An example of its use is at [examples/UsingHTIDClass.ipynb](examples/UsingHTIDClass.ipynb).

## Step 1: Pre-computing Vector_files

The raw token counts for books are not particularly useful. The first step in the pipeline converts books into feature vectors.

- Approach 1: **GloVe** is a pre-trained language model, which seeks to reduce language to a meaningful linear space where similar words are in close proximity to each other. This allows a less rigid similarity approach, where non-exact but similar language is captured.
- Approach 2: **PySRP** is a hash-based representation approach. It projects words based on their hash, and adds the hashes in a text together to represent the text. There is no meaning in proximity between two words - however, the summed vectors retain a fingerprint of the words within them and are comparable. PySRP is more rigid in what it considers 'similar', and is particularly useful for aligning texts that actually are the same.

- Input representation 1: Chunks. This is the primary comparison unit.
- Input representation 2: Full-books. This is mainly for high-level comparisons.
- Input representation 3: Page-level. This is *not* pre-computed for scaling reasons.

Currently, chunk-based vector representations over GloVe and PySRP are computed with [hathi-test-dataset/vectorization.py](https://github.com/massivetexts/hathi-test-dataset/blob/master/vectorization.py). You can also use `--no-srp` or `--no-glove` to exclude one of the vector representations. SRP is faster but requires larger vectors (default is 640-dims vs. 300), while GloVe is smaller and has had slightly better results in the places we've tested it.
This script can be parallelized, in which case you'll want to use [concatenate-vector_files.py](scripts/concatenate-vector_files.py) to patch together the output files, ideally with the 
--build-cache argument.

Paths to various local files should be in `local.yaml`, in the same folder, or `~/.htrc_config.yaml`. GloVe is loaded (and downloaded the first time) using the [gensim downloader](https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html). You can change the path for the models with the `gensim_data_path` parameter in `local.yaml`. If you want to load the model in code, try:

```
from compare_tools.configuration import wem_loader
wem_loader('glove-wiki-gigaword-300')
```

## Step 2: Building an MTAnnoy Index

Annoy is a fast approximate nearest neighbour library from [Erik Bernhardsson](https://github.com/spotify/annoy). MTAnnoy is a wrapper that supports our prefixes, where a book can be included in multiple chunks. It is used to reduce our space of comparisons - this is the 'quick and dirty' matching to give us candidates for matching in slower ways downstream. Since it is meant to be a rough first step, the more permissive properties of Glove are more appropriate here.

The vectorfiles from our previous step can be used to create an MTAnnoy index with [scripts/create-annoy-from-srp.py](scripts/create-annoy-from-srp.py).

```bash
python create-annoy-from-srp.py /data/vectorfiles/all_Glove_testset.bin /data/saddl/annoy/Glove_testset.ann
```

We've written a paper evaluating different parameterizations of Annoy, contact Peter Organisciak for the pre-print.

## Step 3: Exporting Candidates Relationships from MTAnnoy


## Training - Exporting Relationships for Training

### Relationship Sources

Relationships are trained from multiple sources:

- Metadata-based ground truth: exported from high-confidence metadata inferences.
  - Additionally, OCLC data is used
- Generated ('fake') books for relationships that are difficult to find ground truth for.
- GoodReads-based 'similar book' information.
- Annoy-based SIMDIFF information, which is used to train relationships between different books that look similar. This is because RANDIFF (a relationship between two randomly selected works) is too low of a hurdle.

### Training Dataset Creation Step 1: HathiTrust Metadata-based Ground Truth

 See https://github.com/massivetexts/hathi-test-dataset/blob/master/notebooks/MetadataGroundTruth.ipynb
 
### Training Dataset Creation Step 2: OCLC Metadata-based Ground Truth

Data from OCLC's Clasify API is used to augment the SWDE tag, due to imperfect metadata performance with this class elsewhere.
TODO

### Training Dataset Creation Step 3: Fake Books

1. Run [./scripts/FakeBookGeneration.ipynb]. This notebook creates fake books in order to train PARTOF, CONTAINS, and OVERLAPS relationships. This includes fake anthologies, and long books split into multi-volume sets. OVERLAPS include two different anthologies that have a matching sub-unit. This script creates fake EF books and a listing of relationship - they'll need to subsequently be vectorized.
2. Run vectorization code on all the fake books. If parallelized, the output will need to be concatenated into a single vector file also.
3. Crunch stats for training. Since the fake book vectors are likely in a different Vector_file than the one specified in the 'glove_data_path' location (in `local.yaml` or `~/.htrc-config.yaml`), you can point to it at 'glove_fake_data_path' (or any identifier after the first underscore, really). This will be the fallback location for vectors that can't be found in the default location.

### Training Step 1: Processing Similarity Stats

Now we just compare. This is done using the Comparison class, as seen in `ComparisonPipeline.ipynb`. However, there's a script for doing this in an easily parallelizable way, `crunch_stats.py`.

Export a doc of JSON records to compare, minimally `{'left':'..', 'right':'..'}` but optionally with `judgment` and `notes` columns. Sorting by left will make a slight difference downstream. Exporting to the JSON can be done from a DataFrame with `df.sort_values('left').to_dict(orient='records')`. Examples of this can be seen at the bottom of the `MetadataGroundTruth.ipynb` and `FakeBookGeneration.ipynb`.

These JSON records are piped to `scripts/crunch_stats.py`. Example:

- `cat json_stats.json | parallel -j20 -n500 python scripts/crunch_stats.py --outdir /data/save_wherever --save-sim --tfrecord {}`

Note that `n` can't go much higher because there's a system limit on how many system args can be sent to something. If you'd like to split your input into multiple files and have `crunch_stats.py` read from a file, you can provide the filepath to `--input-file` (or `-i`). e.g. here's an example that splits the input into files in `/tmp` then processes those with 20 parallel processes.

```bash
split -l 10000 json_stats.json /tmp/json-stat-chunk
ls /tmp/json-stat-chunk/* | parallel -j20 -n1 python scripts/crunch_stats.py --outdir /data/save_wherever --save-sim --save-wem --tfrecord -i {}`
```

For training, I interwove the 'fake' books with the regular book input.

## Training Step 2: Train classification model.

The most effective approach has been to use a convolution neural network classifier with heavy dropout.

- 
-


## Step 4: Similarity Inference

- In-progress, see [scripts/InferenceOnWork.ipynb]