from htrc_features import FeatureReader, utils
import argparse
import os

def main():

    parser = argparse.ArgumentParser(description='Convert EF files to Parquet compressed with Snappy')
    
    parser.add_argument('--efdir', type=str, default='/data/extracted-features/', help='Location of the EF files')
    parser.add_argument('--outdir', type=str, default='/data/extracted-features-parquet/', help='Output location for parquet files.')

    parser.add_argument('filepaths', type=str, nargs='+',  help='files to convert')
    
    args = parser.parse_args()
    
    for efpath in args.filepaths:
        
        try:
            vol = FeatureReader(os.path.join(args.efdir, efpath)).first()
            path = args.outdir + utils.id_to_rsync(vol.id)
            path, filename = os.path.split(path)

            os.makedirs(path, exist_ok=True)
            a = vol.tokenlist(pos=False).reset_index()
            if a.empty:
                continue
            a[['page', 'token', 'count']].to_parquet(os.path.join(args.outdir, path, filename.replace('.json.bz2', '.parquet')), compression='snappy')
        except:
            raise
            with open('errs.txt', mode='w') as f:
                f.write(efpath + "\n")
            print("Error", efpath)
    
if __name__ == '__main__':
    main()