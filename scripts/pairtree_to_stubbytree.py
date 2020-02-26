import shutil
import os
from htrc_features.utils import _id_encode, id_to_rsync
from compare_tools.utils import StubbytreeResolver
import argparse

def move_htrc(htid, pairtree_root, stubbytree_root, move=False, prune=False, format='json', compression='bz2',
              ignore_suffix=True):
    old = os.path.join(pairtree_root, id_to_rsync(htid))
    new = os.path.join(stubbytree_root,
                       StubbytreeResolver.id_to_stubbytree(None, htid, format=format, compression=compression)
                      )
    olddir = os.path.split(old)[0]
    # Check if new dir exists
    newdir = os.path.split(new)[0]
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    if ignore_suffix:
        all_old_files = os.listdir(olddir)
    else:        
        all_old_files = [old]

    for old_file in all_old_files:
        old_fname = os.path.join(olddir, old_file)
        new_fname = os.path.join(newdir, old_file)

        if move:
            shutil.move(old_fname, new_fname)
        elif prune:
            raise Exception("Can't prune without moving the original file!")
        else:
            shutil.copyfile(old_fname, new_fname)
        
    if prune:
        while True:
            try:
                os.rmdir(olddir)
            except OSError:
                break
            olddir = os.path.split(olddir)[0]
            

def main():
    parser = argparse.ArgumentParser(description='Move provided HTIDs from pairtree to stubbytree, which organizes'
                                     + ' "libid.volid"-style HTIDs in a `libid/volid[::3]/` directory structure. Currently'
                                     + ' ignores the suffix and moves all files at the end of the pairtree'
                                     + 'Test use with GNU Parallel: cat list-of-htids.txt | parallel --eta -n5000 -j20 python scripts/pairtree_to_stubbytree.py --move /data/extracted-features/ /data/extracted-features-stubby/ {}')
    
    parser.add_argument('pairtree_root', type=str, help='Root for pairtree')
    parser.add_argument('stubbytree_root', type=str, help='Root for stubbytree')
    parser.add_argument('--move', action='store_true', help='Move the file and prune its dir. Default just copies.')
    parser.add_argument('htids', type=str, nargs='+', help='One or more HTIDs')

    args = parser.parse_args()
    
    for htid in args.htids:
        try:
            prune = args.move
            move_htrc(htid, args.pairtree_root, args.stubbytree_root, move=args.move, prune=prune)
        except KeyboardInterrupt:
            raise
        except:
            print("Problem with {}, moving on".format(htid))

if __name__ == "__main__":
    main()