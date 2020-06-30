import argparse
import SRP

def main():

    parser = argparse.ArgumentParser(description='Concatenate a number of vector files into single file')

    parser.add_argument('outpath', type=str, help='Place to save the new Vector file.')
    parser.add_argument('--mode', type=str, default='a', help='Write mode for output. By default, appends if the file exists, can be switched to \'w\' to overwrite.')
    parser.add_argument('--build-cache', action='store_true', help='Build a prefix cache after concatenation.')
    parser.add_argument('--no-concat', action='store_true', help="Skip the concatenation, if you're hoping to *just* build cache.")
    parser.add_argument('filepaths', type=str, nargs='*',  help='List of vector files being combines.')
    
    args = parser.parse_args()
    
    if len(filepaths) == 0 and not args.no_concat:
        raise Exception("Nothing to do without input filepaths")
    if args.no_concat and not args.build_cache:
        raise Exception("If you're not concatenating and not building a cache, you're not doing anything.")
    
    if not args.no_concat:
        with SRP.Vector_file(args.filepaths[0], mode="r") as vecf:
            dims = vecf.dims

        with SRP.Vector_file(args.outpath, mode=args.mode, dims=dims) as outf:
            for efpath in args.filepaths:
                print("Concatenating:", efpath)
                outf.concatenate_file(efpath)

    if args.build_cache:
        with SRP.Vector_file(args.outpath, offset_cache=True) as outf:
            print("Building prefix lookup cache")
            outf._build_offset_lookup(sep='-')
    
if __name__ == '__main__':
    main()