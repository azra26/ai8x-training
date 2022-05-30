#!/usr/bin/env python3

"""
Script to generate dataset for Depthnet training and validation from Linnaeus 5 128X128 dataset.
"""

import argparse
import os
import sys
import shutil

def main(source_path, dest_path):  # pylint: disable=too-many-locals
    """
    Main function to iterate over the images in the raw data and generate data samples
    to train/test FaceID model.
    """
    data_dir_list = os.listdir(source_path)
    for i, folder in enumerate(data_dir_list):
        folder_path = os.path.join(source_path, folder)
        prcssd_folder_path = os.path.join(dest_path, folder)
        
        if not os.path.exists(prcssd_folder_path):
            os.makedirs(prcssd_folder_path)
        
        ctr = 0
        for labels in os.listdir(folder_path):
            class_path = os.path.join(folder_path, labels)
            for images in os.listdir(class_path):
                image_path = os.path.join(class_path, images)
                shutil.copy(image_path, os.path.join(prcssd_folder_path, str(ctr)+'_'+str(labels)+'.jpg'))
                print(ctr, image_path)
                ctr+=1


def parse_args():
    """Parses command line arguments"""
    root_folder = os.path.abspath(__file__)
    for _ in range(3):
        root_folder = os.path.dirname(root_folder)

    parser = argparse.ArgumentParser(description='Generate VGGFace-2 dataset to train/test \
                                                  FaceID model.')
    parser.add_argument('-r', '--raw', dest='raw_data_path', type=str,
                        default=os.path.join(root_folder, 'data', 'Linnaeus 5 128X128'),
                        help='Path to raw Linnaeus 5 128X128 dataset folder.')
    parser.add_argument('-d', '--dest', dest='dest_data_path', type=str,
                        default=os.path.join(root_folder, 'data', 'L5_IN'),
                        help='Folder path to store processed data')
    args = parser.parse_args()

    source_path = os.path.join(root_folder, args.raw_data_path)
    dest_path = os.path.join(root_folder, args.dest_data_path)
    return source_path, dest_path


if __name__ == "__main__":
    raw_data_path, dest_data_path = parse_args()
    main(raw_data_path, dest_data_path)
