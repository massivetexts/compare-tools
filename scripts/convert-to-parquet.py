from htrc_features import FeatureReader, utils
import argparse
import os

def main():

    parser = argparse.ArgumentParser(description='Convert EF files to Parquet compressed with Snappy')
    
    parser.add_argument('--efdir', type=str, default='/data/extracted-features/', help='Location of the EF files')
    parser.add_argument('--outdir', type=str, default='/data/extracted-features-parquet/', help='Output location for parquet files.')
    
    parser.add_argument('--chunked', action='store_true', help='Whether to chunk the internal tokenlist.')
    parser.add_argument('--page-ref', action='store_true', help='Store page reference when chunking.')
    parser.add_argument('--chunk-size', type=int, default=5000, help='Word target for chunks.')

    parser.add_argument('--lowercase', action='store_true', help='Lowercase tokens.')
    parser.add_argument('filepaths', type=str, nargs='+',  help='files to convert')
    
    args = parser.parse_args()
    
    for efpath in args.filepaths:
        
        try:
            vol = FeatureReader(os.path.join(args.efdir, efpath)).first()
            path = args.outdir + utils.id_to_rsync(vol.id)
            path, filename = os.path.split(path)

            os.makedirs(path, exist_ok=True)
            token_kwargs=dict(section='body', drop_section=True, pos=False, 
                              case=(not args.lowercase))
            if args.chunked:
                token_kwargs['chunk_target'] = args.chunk_size
                token_kwargs['page_ref'] = args.page_ref
            vol.save_parquet(path, chunked=args.chunked, token_kwargs=token_kwargs)
        except:
            with open('errs.txt', mode='a') as f:
                f.write(efpath + "\n")
            print("Error", efpath)
    
if __name__ == '__main__':
    main()