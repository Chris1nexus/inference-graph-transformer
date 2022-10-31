import subprocess
import argparse
import os 
import shutil


def  save_tiles(slide_path):

    filename = os.path.basename(slide_path)
    FILEID = filename.rsplit('.', maxsplit=1)[0]
    PATCHES_DIR = os.environ['PATCHES_DIR']
    SLIDES_DIR = os.environ['SLIDES_DIR']
    os.makedirs(PATCHES_DIR, exist_ok=True)
    os.makedirs(SLIDES_DIR, exist_ok=True)
    shutil.copy(slide_path, SLIDES_DIR)

    INPUT_PATH = os.path.join(SLIDES_DIR, filename)
    CMD = ['python3', 'src/tile_WSI.py', '-s', '512', '-e', '0', '-j', '16', '-B', '50', '-M', '20', '-o',  PATCHES_DIR, INPUT_PATH]
    subprocess.call(CMD)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    parser.add_argument('--slide_path', type=str, help='path to the WSI slide')
    args = parser.parse_args()


    filename = os.path.basename(args.slide_path)
    FILEID = filename.rsplit('.', maxsplit=1)[0]
    PATCHES_DIR = os.environ['PATCHES_DIR']
    SLIDES_DIR = os.environ['SLIDES_DIR']
    os.makedirs(PATCHES_DIR, exist_ok=True)
    os.makedirs(SLIDES_DIR, exist_ok=True)
    shutil.move(args.slide_path, SLIDES_DIR)

    INPUT_PATH = os.path.join(SLIDES_DIR, filename)


    CMD = ['python3', 'src/tile_WSI.py', '-s', '512', '-e', '0', '-j', '16', '-B', '50', '-M', '20', '-o',  PATCHES_DIR, INPUT_PATH]
    
    subprocess.call(CMD)

