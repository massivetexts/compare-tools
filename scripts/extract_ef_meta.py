import argparse
from compare_tools.hathimeta import get_json_meta
from compare_tools.configuration import config
import sys

def main():

    parser = argparse.ArgumentParser(description='Extract a field from the EF metadata and output CSV to STDout.')

    parser.add_argument('--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--field', type=str, default='page_count', help='Field to extract.')
    parser.add_argument('htids', type=str, nargs='*',  help='List of HTIDs to extract the field from.')
    
    args = parser.parse_args()
    errs = 0
    for i, htid in enumerate(args.htids):
        if htid == 'htid':
            continue
        try:
            meta = get_json_meta(htid, config['parquet_root'])[args.field]
        except:
            errs += 1
            meta = None
        args.outfile.write("{}\t{}\n".format(htid, meta))
        if (i == errs) & (i > 30):
            print('First 30 items all failed. Rather than error-catching, stopping script.')
            break
    args.outfile.close()
    
if __name__ == '__main__':
    main()