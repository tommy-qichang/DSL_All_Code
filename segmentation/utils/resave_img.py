import glob
import PIL.Image
import os

import argparse


def resave_img(path,type):
    files = sorted(glob.glob(os.path.join(path, '**/*.'+type),recursive=True))
    for file in files:
        im = PIL.Image.open(file).convert('RGB')
        im.save(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resave image to png"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="path to the img folder"
    )
    parser.add_argument(
        "--type", type=str, default="png", help="saved image type"
    )

    args = parser.parse_args()
    resave_img(args.path, args.type)



