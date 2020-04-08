from SRP import Vector_file

old = '/data/vectorfiles/all_Glove_testset.bin'
new = '/data/vectorfiles/all_Glove_testset_dedupe.bin'

with Vector_file(old, mode='r', offset_cache=True) as oldvfile:
    with Vector_file(new, mode='w', dims=oldvfile.dims, offset_cache=True) as deduped_vecfile:
        i = 0
        allkeys = list(oldvfile._prefix_lookup.keys())
        for key in allkeys:
            if key == 'last_offset':
                continue
            deduped = []
            for htid, offset in oldvfile.find_prefix(key):
                if htid not in deduped:
                    deduped.append(htid)
                    deduped_vecfile.add_row(htid, offset)
                i += 1
                if (i % 10000) == 0:
                    print(i, end=',')
                    
with Vector_file(new, mode='a', offset_cache=True) as deduped_vecfile:
    deduped_vecfile._build_offset_lookup(sep='-')