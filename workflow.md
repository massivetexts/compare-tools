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

Currently, chunk-based vector representations over GloVe and PySRP are computed with [hathi-test-dataset/vectorization.py](https://github.com/massivetexts/hathi-test-dataset/blob/master/vectorization.py). This script can be parallelized, in which case you'll want to use [concatenate-vector_files.py](scripts/concatenate-vector_files.py), ideally with the 
--build-cache argument to 

## Step 2: Building an MTAnnoy Index

Annoy is a fast approximate nearest neighbour library from [Erik Bernhardsson](https://github.com/spotify/annoy). MTAnnoy is a wrapper that supports our prefixes, where a book can be included in multiple chunks. It is used to reduce our space of comparisons - this is the 'quick and dirty' matching to give us candidates for matching in slower ways downstream. Since it is meant to be a rough first step, the more permissive properties of Glove are more appropriate here.

The vectorfiles from our previous step can be used to create an MTAnnoy index with [scripts/create-annoy-from-srp.py](scripts/create-annoy-from-srp.py).

```
python create-annoy-from-srp.py /data/vectorfiles/all_Glove_testset.bin /data/saddl/annoy/Glove_testset.ann
```

## Step 3: Exporting Candidates Relationships from MTAnnoy