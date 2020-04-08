""" Tools for generating fake EF books.

These functions help in generating new imaginary Extracted Features files, whether by splitting long books into multiple volumes, putting multi-volume works into one, or creating imaginary anthologies.
"""
import hashlib
import numpy as np
import pandas as pd
from htrc_features import Volume
from bpemb import BPEmb
from scipy.spatial.distance import cosine, pdist, cdist, squareform
from compare_tools.hathimeta import clean_description, clean_title

def split_tokenlist(tl, max_start_pages=10, max_end_pages=10):
    # Take a tokenlist and split it into three parts - front matter, backmatter and content. The
    # front and back are randomly cut off - up to fifteen pages.
    # Return meta_json, front_df, center_df, back_df
    a = tl.index.get_level_values('page').unique()
    front_cutoff=np.random.choice(a[a <= max_start_pages]),
    back_cutoff=np.random.choice(a[a >= (a.max()-max_end_pages)])
    
    # For short books
    while front_cutoff > back_cutoff:
        # Move front/back outward a page. Can't just subtract/add, because
        # sometimes pages are missing.
        if front_cutoff > a.min():
            front_cutoff = a[np.roll(a == front_cutoff, -1)][0]
        if back_cutoff < a.max():
            back_cutoff = a[np.roll(a == back_cutoff, 1)][0]
        if (front_cutoff == a.min()) and (back_cutoff == a.max()):
            raise Exception("Can't split this book.")
            
    front = tl.loc[:front_cutoff,]
    center = tl.loc[front_cutoff:back_cutoff,]
    back = tl.loc[back_cutoff:]
    return dict(front_cutoff=front_cutoff, back_cutoff=back_cutoff), front, center, back

def page_range(df):
    ''' Return min and max of index level named "page" '''
    vals = df.index.get_level_values('page')
    return "{}-{}".format(vals.min(), vals.max())

def combine_tokenlist(list_of_dfs):
    ''' Renumbers the pages to be sequential'''
    builder = [list_of_dfs[0]]
    for part in list_of_dfs[1:]:
        a = part.reset_index()
        seq_max =  builder[-1].index.get_level_values('page').max()
        a['page'] = a['page'].add(seq_max+1).subtract(a['page'].min()).astype(int)
        builder.append(a.set_index(['page', 'lowercase']))
    new_book = pd.concat(builder)
    return new_book

def combine_books(ids, style='anthology'):
    ''' Create a fake book from a set of HTIDs'''
    # Use front and back from 1 book, munge the centers
    fake_front = None
    fake_back = None
    centers = []

    # New metadata
    meta = dict(names=[], pub_date=[], source_htids=[], notes="", language="eng", schema_version="1.3", enumeration_chronology="")
    if style=="anthology":
        meta['title'] = "Fake Anthology: "
    elif style=="multivol":
        meta['title'] = "Fake Combined Vol: "

    for i, htid in enumerate(ids):
        vol = Volume(htid, dir='/data/extracted-features-parquet-stubby/', format='parquet', id_resolver='stubbytree')
        tl = vol.tokenlist(case=False, pos=False)
        split_details, front, center, back = split_tokenlist(tl)
        if i == 0:
            fake_front = front
            fake_back = back
            meta['notes'] += "Beginning: {} of {}; ".format(page_range(front), vol.id)
            meta['notes'] += "Ending: {} of {}; ".format(page_range(back), vol.id)
        centers.append(center)
        meta['names'] += vol.author
        meta['title'] += '{}) {}...'.format(i, vol.title[:40])
        meta['source_htids'].append(vol.id)
        meta['notes'] += "{} of {};".format(page_range(center), vol.id)
        if style == "multivol":
            if i == 0:
                meta['enumeration_chronology'] = vol.enumeration_chronology
            else:
                meta['enumeration_chronology'] = meta['enumeration_chronology'].split('-')[0] + '-' + vol.enumeration_chronology.lower().replace('v.','')
        try:
            year = int(vol.year)
            meta['pub_date'].append(year)
        except ValueError:
            pass
    meta['names'] = list(set(meta['names']))
    meta['pub_date'] = np.mean(meta['pub_date'])
    new_tl = combine_tokenlist([front]+centers+[back])
    meta['page_count'] = int(new_tl.index.get_level_values('page').max())
    
    m = hashlib.md5()
    m.update(",".join(meta['source_htids']).encode('utf-8'))
    meta['id'] = "fake.{}".format(m.hexdigest()[:6])
    return meta, new_tl

def save_fake_vol(meta, tokenlist, dir, id_resolver='stubbytree', token_kwargs=dict(case=False, pos=False)):
    vol = Volume(meta['id'], dir=dir, id_resolver=id_resolver, format='parquet', mode='wb')
    vol._tokencounts = tokenlist
    vol.parser.meta = meta
    vol._pagecolname = 'page'
    vol._update_meta_attrs()
    vol.write(vol, token_kwargs=token_kwargs)
    return meta['id']

def pairwise_title_similarity(titles, bpemb_en=None):
    ''' Clean titles and use BPE encodings to compare their similarity'''
    if bpemb_en is None:
        bpemb_en = BPEmb(lang="en")
    # Convert cleaned title to BPE encodings and keep those vectors
    title_vecs = titles.apply(clean_title).apply(bpemb_en.encode_ids).apply(lambda x: bpemb_en.vectors[x].sum(0)).values
    title_vecs = np.vstack(title_vecs)
    title_sims_pairwise = squareform(pdist(title_vecs, metric='cosine'))
    return title_sims_pairwise

def anthology_sample(meta, target=None, max_len=742, max_books=5, min_sim=0.2):
    '''
    meta: dataframe of metadata for the books that you're sampling from.
    target: index number of the seed book. Will randomly sample if none
    max_len: the length to cap the book at. Default (742) is the 95th percentile
        length of real books in the Massive Texts Lab Testing Dataset
    '''
    if target is None:
        target = np.random.randint(meta.shape[0])
    targets = [target]
    page_count = meta.page_count.iloc[targets[0]]
    title_sims = pairwise_title_similarity(meta.title)
    
    # Collect a set of books where none of the titles are similar to the others
    for i in range(1, max_books):
        sims = title_sims[targets].min(axis=0)
        if sims.max() < min_sim:
            break
        next_target = np.random.choice(np.where(sims > min_sim)[0])
        next_target_pages = meta.page_count.iloc[next_target]
        if (page_count + next_target_pages) > max_len:
            continue
        else:
            page_count += next_target_pages
            targets.append(next_target)
    return meta.iloc[targets].htid.tolist()


def consecutive_vol_samples(meta, max_len=742):
    ''' Returns a list of to_combine candidates, based on consecutive runs of volumes. Not complete, and the representative
    for each volume number is randomly sampled.
    '''
    unique = meta.descint.unique()
    runs = []
    current = []
    # Find run possibilities
    for i in unique:
        if len(current) == 0:
            current.append(i)
        else:
            if i == current[-1] + 1:
                current.append(i)
            else:
                if len(current) > 1:
                    runs.append(current)
                current = []
    if len(current) > 1:
        runs.append(current)
       
    recommendations = []
    for run in runs:
        newbook = meta[meta.descint.isin(run)]
        newbook[newbook.page_count.cumsum() < max_len]
        if newbook.shape[0] <= 1:
            continue
        else:
            to_combine = newbook.htid.tolist()
            recommendations.append(to_combine)
            
    return recommendations