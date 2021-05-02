import cv2
import numpy as np
import os
import errno
from os import path
import time
import final_project as stabilization

SRC_FOLDER = "videos/source"
OUT_FOLDER = "videos/output"
VIDEO_EXTENSIONS = set(["mp4", "avi", "mov"])


def main(video_file, output_dir):
    stabilization.smoothing_video(video_file, output_dir)
    print("Done!")


def get_video_file():
    """Get one video file"""
    video_files = []
    output_dirs = []
    subfolders = os.walk(SRC_FOLDER)
    next(subfolders)  # skip the root input folder
    for dirpath, dirnames, fnames in subfolders:

        video_dir = os.path.split(dirpath)[-1]
        output_dir = os.path.join(OUT_FOLDER, video_dir)

        try:
            os.makedirs(output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        print("Processing '" + video_dir + "' folder...")

        video_file = [os.path.join(dirpath, name) for name in fnames]
        # one video per folder
        video_files.append(video_file[0])
        output_dirs.append(output_dir)

    return video_files, output_dirs


if __name__ == "__main__":
    subfolders = os.walk(SRC_FOLDER)
    for dirpath, dirnames, fnames in subfolders:
        for name in fnames:
            if path.splitext(name)[-1][1:].lower() in VIDEO_EXTENSIONS:
                print("Processing '" + name + "...")
                video_files = os.path.join(dirpath, name)
                output_dir = os.path.join(OUT_FOLDER, name.split(".")[0])
                try:
                    os.makedirs(output_dir)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise

                print(video_files)
                print(output_dir)
                main(video_files, output_dir)
