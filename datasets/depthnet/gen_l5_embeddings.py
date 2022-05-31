#!/usr/bin/env python3

"""
Script to generate embeddings for Depthnet training and validation from Linnaeus 5 128X128 dataset.
"""

import argparse
import os
import sys
import pickle
import numpy
import cv2
import torch


class MidasNet(object):
    
    L5 = {
        'mean': 223.74481,
        'std': 226.20352,
        'max': 809.27625,
        'min': -7.033435,
        'norm_max': 2.588516,
        'norm_min': -1.020224
    }
    
    def __init__(self, model_type="MiDaS_small"):
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    def get_output(self, img, out_size=(32,32)):
        """Captures the Midasnet Output"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)

            out_img = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=out_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            out_embed = out_img.flatten()

        out_img = out_img.cpu().numpy()
        out_embed = out_embed.cpu().numpy()
        
        return out_img, out_embed
    
    def get_norm_img(self, img):
        """Normalizes the whole images output with the whole Linnaeus 5 Dataset"""
        std_img = (img - self.L5['mean'])/self.L5['std']
        norm_img = numpy.round(255*(std_img - self.L5['norm_min'])/(self.L5['norm_max'] - self.L5['norm_min']))
        return norm_img


def main(source_path, dest_path):  # pylint: disable=too-many-locals
    """
    Main function to iterate over the images in the extracted data and generate data samples
    and embeddings to train/test FaceID model.
    """

    midas = MidasNet(model_type="MiDaS_small")

    data_dir_list = os.listdir(source_path)
    for i, folder in enumerate(data_dir_list):
        folder_path = os.path.join(source_path, folder)
        prcssd_folder_path = os.path.join(dest_path, folder)

        if not os.path.exists(prcssd_folder_path):
            os.makedirs(prcssd_folder_path)
        
        img_list = []
        embed_list = []
        data_pickle = {}
        
        ctr = 0
        for images in os.listdir(folder_path):
            # Path
            image_path = os.path.join(folder_path, images)
            out_fn = images.replace(".jpg", ".png")
            depth_path = os.path.join(prcssd_folder_path, out_fn)
            
            # Capture Midas Output
            img = cv2.imread(image_path)
            out_img, out_embed = midas.get_output(img, (32,32))
            
            img_list.append(img)
            embed_list.append(out_embed)
            
            # Normalized Image and Save Output
            norm_img = midas.get_norm_img(out_img)
            cv2.imwrite(depth_path, norm_img)

            print(ctr, image_path)
            ctr+=1
        
        data_pickle['img'] = img_list
        data_pickle['embed'] = embed_list
        
        with open(os.path.join(prcssd_folder_path, str(folder)+'.pickle'), 'wb') as handle:
            pickle.dump(data_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        del img_list
        del embed_list
        del data_pickle
    
    del midas


def parse_args():
    """Parses command line arguments"""
    root_folder = os.path.abspath(__file__)
    for _ in range(3):
        root_folder = os.path.dirname(root_folder)

    parser = argparse.ArgumentParser(description='Generate VGGFace-2 dataset to train/test \
                                                  FaceID model.')
    parser.add_argument('-r', '--raw', dest='raw_data_path', type=str,
                        default=os.path.join(root_folder, 'data', 'L5_IN'),
                        help='Path to extracted Linnaeus 5 128X128 dataset folder.')
    parser.add_argument('-d', '--dest', dest='dest_data_path', type=str,
                        default=os.path.join(root_folder, 'data', 'L5_DEPTH'),
                        help='Folder path to store processed data')
    args = parser.parse_args()

    source_path = os.path.join(root_folder, args.raw_data_path)
    dest_path = os.path.join(root_folder, args.dest_data_path)
    return source_path, dest_path


if __name__ == "__main__":
    raw_data_path, dest_data_path = parse_args()
    main(raw_data_path, dest_data_path)
