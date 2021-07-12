from glob import glob
from os.path import basename
import pandas as pd
# import data_loaders
from smart_load import SmartCrop

from multiprocessing import Array

import os
import sys
import skimage
import numpy as np
from cv2 import cv2
from tqdm import tqdm
from scipy.ndimage import zoom
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
from os.path import exists 

def parse_dataset_folder_old(train_dir):
    """Parses the filenames from the given dataset directory into
    name, side of affection, slice number and its label

    args:
        train_dir (str): path to the dataset directory

    Examples of names it can parse correctly:
        # CTR_TRN_009.nii.gz_right_120.png
        # CTR_TRN_007_right_093.png 
    """

    files = glob(train_dir + "*/*")
    files = [x for x in files if "flipped" not in x]
    file_details = []
    for file in files:
        label, fname = file.split("/")[-2], file.split("/")[-1]
        name = fname[:len("CTR_TRN_007")]
        side, slicen = fname[:-4].split("_")[-2], fname[:-4].split("_")[-1]

        file_details.append((name, side, int(slicen), label))
    file_details = pd.DataFrame(file_details)
    file_details.columns = ["name", "side", "slicen", "label"]
    print(file_details)
    return file_details

from os.path import basename
import nibabel as nib

def parse_dataset_folder(train_dir):
    """Parses the filenames from the given dataset directory into
    name, side of affection, slice number and its label

    args:
        train_dir (str): path to the dataset directory

    Examples of names it can parse correctly:
        # CTR_TRN_009.nii.gz_right_120.png
        # CTR_TRN_007_right_093.png 
    """

    files = glob(train_dir + "*/*")
    files = [x for x in files if "flipped" not in x]
    file_details = []
    for file in files:
        label, fname = file.split("/")[-2], file.split("/")[-1]
        name = fname[:len("TRN_0007")]
        side, slicen = fname[:-4].split("_")[-2], fname[:-4].split("_")[-1]

        file_details.append((name, side, int(slicen), label))
    file_details = pd.DataFrame(file_details)
    file_details.columns = ["name", "side", "slicen", "label"]
    print(file_details)
    return file_details

def parse_iccv_dataset_folder(train_dir):
    """Parses the filenames from the given dataset directory into
    name, side of affection, slice number and its label

    args:
        train_dir (str): path to the dataset directory

    Examples of names it can parse correctly:
        # CTR_TRN_009.nii.gz_right_120.png
        # CTR_TRN_007_right_093.png 
    """

    files = glob(train_dir + "*/*")
    file_details = []
    for file in files:
        label, name = file.split("/")[-2], file.split("/")[-1]
        imgs = sorted([basename(x) for x in glob(file + "/*")])
        for img in imgs:
            slicen = img.split(".")[0]
            file_details.append((name, 'left', int(slicen), label))
            file_details.append((name, 'right', int(slicen), label))
    file_details = pd.DataFrame(file_details)
    
    file_details.columns = ["name", "side", "slicen", "label"]
    print(file_details)
    return file_details

from os.path import basename
import nibabel as nib


def show_vol_info():
    nii_root_path = "/home/sentic/Documents/data/imageclef2021/data/"
    nii_paths = glob(nii_root_path + "*.nii.gz")

    for nii_path in sorted(nii_paths):
        print(basename(nii_path))
        vol = nib.load(nii_path)
        header = vol.header
        # print(header.get_data_shape()[2])
        print(header.get_zooms())

def generate_dataset_details(metadata_path, root_vol_path):
    metadata = pd.read_csv(metadata_path)

    file_details = []
    for row in metadata.iterrows():
        name = row["FileName"]
        label = row["TypeOfTB"]
        vol_path = root_vol_path + "/" + name

        header = nib.load(vol_path).header
        sliceno = header.get_data_shape()[2]
        file_details.extend([[name, side, x, label] for x in range(sliceno) for side in ["left", "right"]])
    "-----------------------------------------------------------------"   


import joblib
class createPngDataset_minivol():
    """CLEF dataset."""
    from smart_load import SmartCrop

    def __init__(self, csv_file, root_dir, out_dir=None, transform=None, reshape=(96, 256, 128), multiprocess=True, preload_path=None, save_fname="train.jl"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(os.path.join(root_dir, csv_file))
        self.images = []
        self.root_dir = root_dir
        self.transform = transform
        self.multiprocess = multiprocess
        self.out_dir = out_dir

        self.crop_perct = (0, 0)
        self.reshape = reshape
        self.scrop = self.SmartCrop(output_size=(256, 512), maskthreshold=160, np_array=True)

        self.process_metadata()
        if preload_path:
            print("Loading...")
            self.images = joblib.load(self.out_dir + preload_path)
        else:
            self.load_all()
            print(len(self.images))
            joblib.dump(self.images, self.out_dir + save_fname)
        print("Loaded: " + str(self.__len__()))

    def __len__(self):
        return len(self.metadata) * self.reshape[0]

    def __getitem__(self, idx):
        try:
            return self.images[idx]
        except:
            print("Error: " + str(len(self.images)) + "; " + str(idx))
            return self.images[0]

    def save_image(self, img, label, name, side, slicen, dir_name="test_set_rest"):
        # img = np.rollaxis(img, 0, 3)
        img = Image.fromarray(img)  # .convert('RGB')

        label_dir_path = self.out_dir + dir_name + "/"
        if not os.path.exists(label_dir_path):
            os.makedirs(label_dir_path)
        Path(label_dir_path).mkdir(parents=True, exist_ok=True)
        # side ="left" if "left" in name else "right"
        img.save(label_dir_path + name + "_" + side + "_" + str(slicen) + ".png")

    def process_image(self, i):
        i = 2 * i

        # print(self.metadata.iloc[i, 0])
        # print(self.metadata.iloc[i])
        # print(self.metadata)

        img_path = os.path.join(self.root_dir, self.metadata.iloc[i, 0])

        # Read an image with OpenCV
        # image = sitk.ReadImage(img_path)
        # inimg_raw = sitk.GetArrayFromImage(image)
        # directions = np.asarray(image.GetDirection())
        # if len(directions) == 9:
        #     tvolslices = np.flip(inimg_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
        # del image
        # del directions

        pics = sorted(glob(img_path + "/*"))
        print(img_path)
        print(pics[0])

        tvolslices = []
        for pic in pics:
            pic = cv2.imread(pic)
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            tvolslices.append(pic)
        tvolslices = np.stack(tvolslices)
    
        # resize to depth
        trnsfs = (self.reshape[0] / tvolslices.shape[0], 1, 1)
        tvolslices = zoom(tvolslices, trnsfs)

        # smart crop
        new_slices = []
        for j in range(tvolslices.shape[0]):
            new_slices.append(self.scrop.call2(tvolslices[j, :, :], thresh=150))
        tvolslices = np.asarray(new_slices)

        # divide left/right + to image
        slices = []
        for j in [0, 1]:
            labels = self.metadata.iloc[i + j, 2:]
            labels = np.array(labels)
            labels = labels.astype('float32').T
            if not j:  # right # reversed not !!!!!!!!!!!!!!!!!
                # print(str(j) + ": " + str(tvolslices.shape))
                img = self.to_image(tvolslices[:, :, :256], self.T1_WINDOW_LEVEL)
                for l in range(img.shape[0]):
                    #-----------------
                    minivol = img[l, :, :]
                    shape = img.shape
                    img_prev = img[l+2, :, :] if l+2<shape[0] else img[l, :, :]
                    img_next = img[l-2, :, :] if l-2>=0 else img[l, :, :]
                    minivol = np.stack([img_prev, minivol, img_next], axis=0)
                    minivol = np.rollaxis(minivol, 0, 3)
                    #-----------------

                    slices.append((minivol, labels, basename(img_path) + "_left"))
                    self.save_image(minivol, labels[0], basename(img_path), "left", l)
                # print(str(j) + ": " + str(tvolslices.shape))
            else:  # left
                # print(str(j) + ": " + str(tvolslices.shape))
                img = self.to_image(tvolslices[:, :, 256:], self.T1_WINDOW_LEVEL)
                for l in range(img.shape[0]):
                    #-----------------
                    minivol = img[l, :, :]
                    shape = img.shape
                    img_prev = img[l+2, :, :] if l+2<shape[0] else img[l, :, :]
                    img_next = img[l-2, :, :] if l-2>=0 else img[l, :, :]
                    minivol = np.stack([img_prev, minivol, img_next], axis=0)
                    minivol = np.rollaxis(minivol, 0, 3)

                    #-----------------
                    slices.append((minivol, labels, basename(img_path) + "_right"))
                    self.save_image(minivol, labels[0], basename(img_path), "right", l)

                # slices.append((tvolslices[:, :, 256:], labels, basename(img_path) + "_right"))
                # print(str(j) + ": " + str(tvolslices.shape))
        
        return slices

    def process_metadata(self):
        # process metadata file
        new = []
        for i in range(self.metadata.shape[0]):
            name = self.metadata.iloc[i, 0][:] # why 15?
            labels = self.metadata.iloc[i, 1:].tolist()
            orient = 0 if 'left' in self.metadata.iloc[i, 0] else 1
            name2 = name.replace("_left", "").replace("_right", "")

            new.append([name2, orient, *labels])
        new = pd.DataFrame(new)
        # new = new.drop_duplicates()
        # new = new.reset_index()
        self.metadata = new
        print(self.metadata)

    def load_all(self):
        from multiprocessing import Pool

        print("Loading data...")
        if self.multiprocess:
            pool = Pool(16)                        # Create a multiprocessing Pool

            # print(self.metadata.shape)
            # process data_inputs iterable with pool
            self.images = list(tqdm(pool.imap(self.process_image, range(
                int(self.metadata.shape[0] / 2))), total=int(self.metadata.shape[0] / 2)))
            pool.close()
            pool.join()
            self.images = [item for sublist in self.images for item in sublist]
            print(len(self.images))
            # pool.map(self.process_image, range(int(self.metadata.shape[0] / 2)))
        else:
            # print(self.metadata.shape[0])
            for i in tqdm(range(0, self.metadata.shape[0], 2)):
                # print(i)
                self.process_image(i)

class createPngDataset_minivol3():
    """Transforms the dataset from a grayscale image to 3 channel
     color image by appending to the middle channel the original image,
     to the previous channel the image 2 slices below and to the next channel 
     the image 2 slices above (creating a mini volume). I chose 2 slices as """

    def __init__(self, data, root_dir, out_dir, shape=(256, 256), multiprocess=True, dir_name="train_default"):
        """
        Args:
            data (panads DataFrame): parsed files (should include name, label, side and slice number)
            root_dir (string): Directory with all the images.
            out_dir (callable, optional): Directory to save the new dataset.
            shape (tuple): The shape of the resulting transformation.
            multiprocess (boolean): Switch to use multiprocess (speeds up the processing considerably)
            dir_name (str): Name of the resulting directory.
        """
        self.data = data
        self.root_dir = root_dir
        self.out_dir = out_dir

        self.multiprocess = multiprocess
        self.reshape = shape
        self.dir_name = dir_name

        # internal
        self.scrop = SmartCrop(output_size=(
            256, 512), maskthreshold=160, np_array=True)

        # globals
        self.images = []

    def process_image(self, group):
        try:
            name = group.iloc[0]["name"]
            img_path = os.path.join(self.root_dir, group["label"].to_list()[0] + "/" + name)

            # label = group["label"].to_list()[0]
            # side = "right"
            # slicen = 0
            # label_dir_path = self.out_dir + self.dir_name + "/" + str(label)
            # if exists(label_dir_path + "/" + name + "_" + side + "_" + str(slicen) + ".png"):
            #     return

            # Read an image with OpenCV
            paths = [(x, int(basename(x).split(".")[0])) for x in glob(img_path + "/*")]
            pics = [x[0] for x in sorted(paths, key=lambda x: x[1])]
            tvolslices = []
            for pic in pics:
                pic = cv2.imread(pic)
                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                try:
                    tvolslices.append(pic)
                except:
                    pass
            tvolslices = np.stack(tvolslices)

            # resize to depth
            # trnsfs = (self.reshape[0] / tvolslices.shape[0], 1, 1)
            trnsfs = (self.reshape[0] / tvolslices.shape[0], self.reshape[1] / tvolslices.shape[1], self.reshape[2] / tvolslices.shape[2])

            tvolslices = zoom(tvolslices, trnsfs)
            # print(tvolslices.shape)

            # smart crop
            new_slices = []
            for j in range(tvolslices.shape[0]):
                new_slices.append(self.scrop.call2(
                    tvolslices[j, :, :], thresh=160))
            tvolslices = np.asarray(new_slices)

            label = group["label"].to_list()[0]
            for side in ["left", "right"]:
                for i in range(tvolslices.shape[0]):
                    slicen = i

                    label_dir_path = self.out_dir + self.dir_name + "/" + str(label)
                    if exists(label_dir_path + "/" + name + "_" + side + "_" + str(slicen) + ".png"):
                        continue
                    
                    try:
                        img = tvolslices[slicen, :, :]
                    except:
                        print("Exception at: " + label_dir_path + "/" + name + "_" + side + "_" + str(slicen) + ".png")
                        print(slicen)
                        print(tvolslices.shape)
                        print(len(paths))
                        print(len(pics))
                        print(img_path + "/*")
                        print(group.shape)
                        print(group["label"].to_list())
                        # print(pics)
                        continue
                    shape = tvolslices.shape
                    img_prev = tvolslices[slicen+2, :, :] if slicen + \
                        2 < shape[0] else tvolslices[slicen, :, :]
                    img_next = tvolslices[slicen-2, :, :] if slicen - \
                        2 >= 0 else tvolslices[slicen, :, :]

                    img = np.stack([img_prev, img, img_next], axis=0)

                    if side == 'left':
                        img = img[:, :, :256]
                    elif side == 'right':
                        img = img[:, :, 256:]
                    else:
                        print("SIDE IS WRONG!")
                        return

                    img = np.rollaxis(img, 0, 3)
                    img = Image.fromarray(img)  # .convert('RGB')

                    if not os.path.exists('my_folder'):
                        os.makedirs('my_folder')
                    label_dir_path = self.out_dir + self.dir_name + "/" + str(label)
                    Path(label_dir_path).mkdir(parents=True, exist_ok=True)
                    # print(label_dir_path + "/" + name + "_" +
                    #          side + "_" + str(slicen) + ".png")
                    img.save(label_dir_path + "/" + name + "_" +
                            side + "_" + str(slicen) + ".png")
        except Exception as e:
            print(e)
            print("BIG ERROR!")

    def generate_train(self):
        from multiprocessing import Pool

        groups = self.data.groupby(["name", "label"])
        groups = [x[1] for x in groups]

        print("Loading data...")
        if self.multiprocess:
            # Create a multiprocessing Pool
            pool = Pool(16)
            # process data_inputs iterable with pool
            self.images = list(
                tqdm(pool.imap(self.process_image, groups), total=int(len(groups))))
            pool.close()
            pool.join()
        else:
            for group in tqdm(groups):
                self.process_image(group)


class createPngDataset_minivol2():
    """Transforms the dataset from a grayscale image to 3 channel
     color image by appending to the middle channel the original image,
     to the previous channel the image 2 slices below and to the next channel 
     the image 2 slices above (creating a mini volume). I chose 2 slices as """

    def __init__(self, data, root_dir, out_dir, shape=(256, 256), multiprocess=True, dir_name="train_default"):
        """
        Args:
            data (panads DataFrame): parsed files (should include name, label, side and slice number)
            root_dir (string): Directory with all the images.
            out_dir (callable, optional): Directory to save the new dataset.
            shape (tuple): The shape of the resulting transformation.
            multiprocess (boolean): Switch to use multiprocess (speeds up the processing considerably)
            dir_name (str): Name of the resulting directory.
        """
        self.data = data
        self.root_dir = root_dir
        self.out_dir = out_dir

        self.multiprocess = multiprocess
        self.reshape = shape
        self.dir_name = dir_name

        # internal
        self.scrop = SmartCrop(output_size=(
            256, 512), maskthreshold=160, np_array=True)

        # globals
        self.images = []

    def process_image(self, group):
        try:
            name = group.iloc[0]["name"]
            img_path = os.path.join(self.root_dir, group["label"].to_list()[0] + "/" + name)

            # Read an image with OpenCV
            paths = [(x, int(basename(x).split(".")[0])) for x in glob(img_path + "/*")]
            pics = [x[0] for x in sorted(paths, key=lambda x: x[1])]
            tvolslices = []
            for pic in pics:
                pic = cv2.imread(pic)
                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                try:
                    tvolslices.append(pic)
                except:
                    pass
            tvolslices = np.stack(tvolslices)

            # resize to depth
            trnsfs = (self.reshape[0] / tvolslices.shape[0], 1, 1)
            tvolslices = zoom(tvolslices, trnsfs)

            # smart crop
            new_slices = []
            for j in range(tvolslices.shape[0]):
                new_slices.append(self.scrop.call2(
                    tvolslices[j, :, :], thresh=160))
            tvolslices = np.asarray(new_slices)

            for i, row in group.iterrows():
                slicen = row["slicen"]
                side = row["side"]
                label = row["label"]

                label_dir_path = self.out_dir + self.dir_name + "/" + str(label)
                if exists(label_dir_path + "/" + name + "_" + side + "_" + str(slicen) + ".png"):
                    continue
                
                try:
                    img = tvolslices[slicen, :, :]
                except:
                    print("Exception at: " + label_dir_path + "/" + name + "_" + side + "_" + str(slicen) + ".png")
                    print(slicen)
                    print(tvolslices.shape)
                    print(len(paths))
                    print(len(pics))
                    print(img_path + "/*")
                    print(group.shape)
                    print(group["label"].to_list())
                    # print(pics)
                    continue
                shape = tvolslices.shape
                img_prev = tvolslices[slicen+2, :, :] if slicen + \
                    2 < shape[0] else tvolslices[slicen, :, :]
                img_next = tvolslices[slicen-2, :, :] if slicen - \
                    2 >= 0 else tvolslices[slicen, :, :]

                img = np.stack([img_prev, img, img_next], axis=0)

                if side == 'left':
                    img = img[:, :, :256]
                elif side == 'right':
                    img = img[:, :, 256:]
                else:
                    print("SIDE IS WRONG!")
                    return

                img = np.rollaxis(img, 0, 3)
                img = Image.fromarray(img)  # .convert('RGB')

                if not os.path.exists('my_folder'):
                    os.makedirs('my_folder')
                label_dir_path = self.out_dir + self.dir_name + "/" + str(label)
                Path(label_dir_path).mkdir(parents=True, exist_ok=True)
                # print(label_dir_path + "/" + name + "_" +
                #          side + "_" + str(slicen) + ".png")
                img.save(label_dir_path + "/" + name + "_" +
                        side + "_" + str(slicen) + ".png")
        except:
            print("BIG ERROR!")

    def generate_train(self):
        from multiprocessing import Pool

        groups = self.data.groupby(["name", "label"])
        groups = [x[1] for x in groups]

        print("Loading data...")
        if self.multiprocess:
            # Create a multiprocessing Pool
            pool = Pool(16)
            # process data_inputs iterable with pool
            self.images = list(
                tqdm(pool.imap(self.process_image, groups), total=int(len(groups))))
            pool.close()
            pool.join()
        else:
            for group in tqdm(groups):
                self.process_image(group)



class createNiiDataset_minivol():
    """Transforms the dataset from a grayscale image to 3 channel
     color image by appending to the middle channel the original image,
     to the previous channel the image 2 slices below and to the next channel 
     the image 2 slices above (creating a mini volume). I chose 2 slices as """

    def __init__(self, data, root_dir, out_dir, shape=(256, 256), multiprocess=True, dir_name="train_default"):
        """
        Args:
            data (panads DataFrame): parsed files (should include name, label, side and slice number)
            root_dir (string): Directory with all the images.
            out_dir (callable, optional): Directory to save the new dataset.
            shape (tuple): The shape of the resulting transformation.
            multiprocess (boolean): Switch to use multiprocess (speeds up the processing considerably)
            dir_name (str): Name of the resulting directory.
        """
        self.data = data
        self.root_dir = root_dir
        self.out_dir = out_dir

        self.multiprocess = multiprocess
        self.reshape = shape
        self.dir_name = dir_name

        # internal
        self.T1_WINDOW_LEVEL = (2500, -1000)
        self.scrop = SmartCrop(output_size=(
            256, 512), maskthreshold=-300, np_array=True)

        # globals
        self.images = []

    def to_image(self, volume, T1_WINDOW_LEVEL):
        img = sitk.GetImageFromArray(volume)
        img = sitk.Cast(sitk.IntensityWindowing(img,
                                                windowMinimum=T1_WINDOW_LEVEL[1] -
                                                T1_WINDOW_LEVEL[0] / 2.0,
                                                windowMaximum=T1_WINDOW_LEVEL[1] + T1_WINDOW_LEVEL[0] / 2.0),
                        sitk.sitkUInt8)
        img = sitk.GetArrayFromImage(img)
        return img

    def process_image(self, group):
        name = group.iloc[0]["name"]
        img_path = os.path.join(self.root_dir, name + ".nii.gz")

        # Read an image with OpenCV
        image = sitk.ReadImage(img_path)
        inimg_raw = sitk.GetArrayFromImage(image)
        directions = np.asarray(image.GetDirection())
        if len(directions) == 9:
            tvolslices = np.flip(inimg_raw, np.where(
                directions[[0, 4, 8]][::-1] < 0)[0])
        del image
        del directions

        # smart crop
        new_slices = []
        for j in range(tvolslices.shape[0]):
            new_slices.append(self.scrop.call2(
                tvolslices[j, :, :], thresh=-1600))
        tvolslices = np.asarray(new_slices)

        tvolslices = self.to_image(tvolslices[:, :, :], self.T1_WINDOW_LEVEL)
        for i, row in group.iterrows():
            slicen = row["slicen"]
            side = row["side"]
            label = row["label"]

            img = tvolslices[slicen, :, :]
            shape = tvolslices.shape
            img_prev = tvolslices[slicen+2, :, :] if slicen + \
                2 < shape[0] else tvolslices[slicen, :, :]
            img_next = tvolslices[slicen-2, :, :] if slicen - \
                2 >= 0 else tvolslices[slicen, :, :]

            img = np.stack([img_prev, img, img_next], axis=0)

            if side == 'left':
                img = img[:, :, :256]
            elif side == 'right':
                img = img[:, :, 256:]
            else:
                print("SIDE IS WRONG!")
                return

            img = np.rollaxis(img, 0, 3)
            img = Image.fromarray(img)  # .convert('RGB')

            if not os.path.exists('my_folder'):
                os.makedirs('my_folder')
            label_dir_path = self.out_dir + self.dir_name + "/" + str(label)
            Path(label_dir_path).mkdir(parents=True, exist_ok=True)
            img.save(label_dir_path + "/" + name + "_" +
                     side + "_" + str(slicen) + ".png")

    def generate_train(self):
        from multiprocessing import Pool

        groups = self.data.groupby("name")
        groups = [x[1] for x in groups]

        print("Loading data...")
        if self.multiprocess:
            # Create a multiprocessing Pool
            pool = Pool(16)
            # process data_inputs iterable with pool
            self.images = list(
                tqdm(pool.imap(self.process_image, groups), total=int(len(groups))))
            pool.close()
            pool.join()
        else:
            for group in tqdm(groups):
                self.process_image(group)


class createNiiDataset_windows():
    """Transforms the dataset from a grayscale image to 3 channel
     color image by assigning to each channel a different HU to grayscale
     transformation (using different window level and range"""

    def __init__(self, data, root_dir, out_dir, shape=(256, 256), multiprocess=True, dir_name="train_default"):
        """
        Args:
            data (panads DataFrame): parsed files (should include name, label, side and slice number)
            root_dir (string): Directory with all the images.
            out_dir (callable, optional): Directory to save the new dataset.
            shape (tuple): The shape of the resulting transformation.
            multiprocess (boolean): Switch to use multiprocess (speeds up the processing considerably)
            dir_name (str): Name of the resulting directory.
        """
        self.data = data
        self.root_dir = root_dir
        self.out_dir = out_dir

        self.multiprocess = multiprocess
        self.reshape = shape
        self.dir_name = dir_name

        # internal
        self.T1_WINDOW_LEVEL = [(2500, -1000), (350, 40), (500, -600)]
        self.scrop = SmartCrop(output_size=(
            256, 512), maskthreshold=-300, np_array=True)

        # globals
        self.images = []

    def to_image(self, volume, T1_WINDOW_LEVEL):
        img = sitk.GetImageFromArray(volume)
        img = sitk.Cast(sitk.IntensityWindowing(img,
                                                windowMinimum=T1_WINDOW_LEVEL[1] -
                                                T1_WINDOW_LEVEL[0] / 2.0,
                                                windowMaximum=T1_WINDOW_LEVEL[1] + T1_WINDOW_LEVEL[0] / 2.0),
                        sitk.sitkUInt8)
        img = sitk.GetArrayFromImage(img)
        return img

    def process_image(self, group):
        name = group.iloc[0]["name"]
        img_path = os.path.join(self.root_dir, name + ".nii.gz")

        # Read an image with OpenCV
        image = sitk.ReadImage(img_path)
        inimg_raw = sitk.GetArrayFromImage(image)
        directions = np.asarray(image.GetDirection())
        if len(directions) == 9:
            tvolslices = np.flip(inimg_raw, np.where(
                directions[[0, 4, 8]][::-1] < 0)[0])
        del image
        del directions

        # smart crop
        new_slices = []
        for j in range(tvolslices.shape[0]):
            new_slices.append(self.scrop.call2(
                tvolslices[j, :, :], thresh=-300))
        tvolslices = np.asarray(new_slices)

        tvolslices1 = self.to_image(
            tvolslices[:, :, :], self.T1_WINDOW_LEVEL[0])
        tvolslices2 = self.to_image(
            tvolslices[:, :, :], self.T1_WINDOW_LEVEL[1])
        tvolslices3 = self.to_image(
            tvolslices[:, :, :], self.T1_WINDOW_LEVEL[2])

        for i, row in group.iterrows():
            slicen = row["slicen"]
            side = row["side"]
            label = row["label"]

            img = tvolslices1[slicen, :, :]
            img_prev = tvolslices2[slicen, :, :]
            img_next = tvolslices3[slicen, :, :]

            img = np.stack([img_prev, img, img_next], axis=0)

            if side == 'left':
                img = img[:, :, :256]
            elif side == 'right':
                img = img[:, :, 256:]
            else:
                print("SIDE IS WRONG!")
                return

            img = np.rollaxis(img, 0, 3)
            img = Image.fromarray(img)  # .convert('RGB')

            if not os.path.exists('my_folder'):
                os.makedirs('my_folder')
            label_dir_path = self.out_dir + self.dir_name + "/" + str(label)
            Path(label_dir_path).mkdir(parents=True, exist_ok=True)
            img.save(label_dir_path + "/" + name + "_" +
                     side + "_" + str(slicen) + ".png")

    def generate_train(self):
        from multiprocessing import Pool

        groups = self.data.groupby("name")
        groups = [x[1] for x in groups]

        print("Loading data...")
        if self.multiprocess:
            # Create a multiprocessing ')Pool
            pool = Pool(16)
            # process data_inputs iterable with pool
            self.images = list(
                tqdm(pool.imap(self.process_image, groups), total=int(len(groups))))
            pool.close()
            pool.join()
        else:
            for group in tqdm(groups):
                self.process_image(group)


class createNiiDataset2():
    """CLEF dataset."""
    from smart_load import SmartCrop

    def __init__(self, data, root_dir, out_dir, noHU=False, shape=(256, 256), multiprocess=True, dir_name="train_default"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.root_dir = root_dir
        self.out_dir = out_dir

        self.noHU = noHU
        self.multiprocess = multiprocess
        self.reshape = shape
        self.dir_name = dir_name

        # internal
        self.min_max = (-600, 100)
        self.T1_WINDOW_LEVEL = [(2500, -1000), (350, 40), (500, -600)]
        self.scrop = self.SmartCrop(output_size=(256, 512), maskthreshold=-300, np_array=True)

        # globals
        self.images = []
        
    def to_image(self, volume, T1_WINDOW_LEVEL):
        img = sitk.GetImageFromArray(volume)
        img = sitk.Cast(sitk.IntensityWindowing(img,
                                                windowMinimum=T1_WINDOW_LEVEL[1] - T1_WINDOW_LEVEL[0] / 2.0,
                                                windowMaximum=T1_WINDOW_LEVEL[1] + T1_WINDOW_LEVEL[0] / 2.0),
                        sitk.sitkUInt8)
        img = sitk.GetArrayFromImage(img)
        return img

    def process_image(self, group):
        name = group.iloc[0]["name"]
        img_path = os.path.join(self.root_dir, name + ".nii.gz")

        # Read an image with OpenCV
        image = sitk.ReadImage(img_path)
        inimg_raw = sitk.GetArrayFromImage(image)
        directions = np.asarray(image.GetDirection())
        if len(directions) == 9:
            tvolslices = np.flip(inimg_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
        del image
        del directions

        # smart crop
        new_slices = []
        for j in range(tvolslices.shape[0]):
            new_slices.append(self.scrop.call2(tvolslices[j, :, :], thresh=-1600))
        tvolslices = np.asarray(new_slices)

        tvolslices1 = self.to_image(tvolslices[:, :, :], self.T1_WINDOW_LEVEL[0])
        tvolslices2 = self.to_image(tvolslices[:, :, :], self.T1_WINDOW_LEVEL[0])
        tvolslices3 = self.to_image(tvolslices[:, :, :], self.T1_WINDOW_LEVEL[0])

        for i, row in group.iterrows():
            slicen = row["slicen"]
            side = row["side"]
            label = row["label"]
            

            img = tvolslices1[slicen, :, :]
            img_prev = tvolslices2[slicen, :, :]
            img_next = tvolslices3[slicen, :, :]

            img = np.stack([img_prev, img, img_next], axis=0)
            # print(np.max(img))
            # print(np.min(img))
            # print(name)

            if side == 'left':
                img = img[:, :, :256]
            elif side == 'right':
                img = img[:, :, 256:]
            else:
                print("SIDE IS WRONG!")
                return

            img = np.rollaxis(img, 0, 3)
            img = Image.fromarray(img)#.convert('RGB')

            if not os.path.exists('my_folder'):
                os.makedirs('my_folder')
            label_dir_path = self.out_dir + self.dir_name + "/" + str(label)
            Path(label_dir_path).mkdir(parents=True, exist_ok=True)
            img.save(label_dir_path + "/" + name + "_" + side + "_" + str(slicen) + ".png")


    def generate_train(self):
        from multiprocessing import Pool

        groups = self.data.groupby("name")
        groups = [x[1] for x in groups]

        print("Loading data...")
        if self.multiprocess:
            pool = Pool(16)                        # Create a multiprocessing Pool
            # process data_inputs iterable with pool
            self.images = list(tqdm(pool.imap(self.process_image, groups), total=int(len(groups))))
            pool.close()
            pool.join()
        else:
            for group in tqdm(groups):
                self.process_image(group)


if __name__ == "__main__":
    # train_dir = "/home/sentic/Documents/data/Madu/Medical/LUNGS/EXPERIMENTS/data/train_temp/"
    # data_dir = "/home/sentic/Documents/md/data/tuberculosis/images/"
    # out_dir = "/home/sentic/Documents/data/Madu/Medical/LUNGS/EXPERIMENTS/data/"

    # train_dir = "/data/train_man/"
    # data_dir = "/data/data/"
    # out_dir = "/data/"

    # train_dir = "/home/sentic/storage2/iccv/train/"
    # data_dir = "/home/sentic/storage2/iccv/train_minivol/"
    # out_dir = "/home/sentic/storage2/iccv/train_minivol/"

    train_dir = "/data/train/"
    data_dir = "/data/train_minivol/"
    out_dir = "/code/train_minivol/"

    train_dir = "/data/val/"
    data_dir = "/data/val_minivol/"
    out_dir = "/code/val_minivol/"
    # # minivols
    # data = parse_dataset_folder(train_dir)
    # dataset_gen = createNiiDataset_minivol(data, data_dir, out_dir, shape=(
    #     256, 256), multiprocess=True, dir_name="train_minivol")
    # dataset_gen.generate_train()

    # minivols
    data = parse_iccv_dataset_folder(train_dir)
    dataset_gen = createPngDataset_minivol3(data, train_dir, out_dir, shape=(96,
        256, 256), multiprocess=True, dir_name="val_minivol")
    dataset_gen.generate_train()

    # print(glob("/home/sentic/storage2/iccv/train/covid/ct_scan_10/*"))
    # print(len(glob("/home/sentic/storage2/iccv/train/covid/ct_scan_10/*")))

    # # windows
    # data = parse_dataset_folder(train_dir)
    # dataset_gen = createNiiDataset2(data, data_dir, out_dir, shape=(
    #     256, 256), multiprocess=True, dir_name="val_naive")
    # dataset_gen.generate_train()
