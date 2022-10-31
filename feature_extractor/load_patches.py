
import os, glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    wsi_slides_paths = []


    def r(dirpath):
        for file in os.listdir(dirpath):
            path = os.path.join(dirpath, file)
            if os.path.isfile(path) and file.endswith(".svs"):
                wsi_slides_paths.append(path)
            elif os.path.isdir(path):
                r(path)
    def r(dirpath):
        for path in glob.glob(os.path.join(dirpath, '*','*.svs') ):#os.listdir(dirpath):
            if os.path.isfile(path):
                wsi_slides_paths.append(path)    
    def r(dirpath):
        for path in glob.glob(os.path.join(dirpath, '*', '*', '*.jpeg') ):#os.listdir(dirpath):
            if os.path.isfile(path):
                wsi_slides_paths.append(path)                             
    r(args.data_path)
    with open('all_patches.csv', 'w') as f:
        for filepath in wsi_slides_paths:
            f.write(f'{filepath}\n')




if __name__ == "__main__":
    main()
