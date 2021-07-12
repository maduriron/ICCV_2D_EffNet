import joblib
import os
import numpy as np
import pandas as pd
import skimage.io as io
from PIL import Image
from os.path import basename
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from multiprocessing import Array

import sys
from tqdm import tqdm
import skimage
from cv2 import cv2
from scipy.ndimage import zoom
import SimpleITK as sitk
from pathlib import Path

import augmentations as augmentations
# sys.path.append("../pytorch-template/")

class CLEFDataset(Dataset):
    """CLEF dataset."""

    def __init__(self, csv_file, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(os.path.join(root_dir, csv_file), header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.metadata.iloc[idx, 0])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        if self.train:
            labels = self.metadata.iloc[idx, 1:]
            labels = np.array(labels)
            labels = labels.astype('float32').T
            return image, labels
        else:
            return basename(img_path), image


class CLEFDataLoader():
    """
    CLEF data loading class using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, num_workers=4):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]),
        transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = CLEFDataset("train.csv", data_dir, transformations)
        self.val_dataset = CLEFDataset("val.csv", data_dir, transformations)

    def get_train_loader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return train_loader

    def get_val_loader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True, drop_last=False)
        return val_loader


class CLEFDataset_albumentations():
    """CLEF dataset."""

    def __init__(self, csv_file, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(os.path.join(root_dir, csv_file), header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.metadata.iloc[idx, 0])
        # Read an image with OpenCV
        image = cv2.imread(img_path)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.train:
            labels = self.metadata.iloc[idx, 1:]
            labels = np.array(labels)
            labels = labels.astype('float32').T
            return image, labels
        else:
            return basename(img_path), image


class CLEFDataLoader_albumentations():
    """
    CLEF data loading class using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, num_workers=6):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]),
        train_transformations = augmentations.strong_aug()
        val_transformations = augmentations.no_aug()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = CLEFDataset_albumentations("train.csv", data_dir, train_transformations)
        self.val_dataset = CLEFDataset_albumentations("val.csv", data_dir, val_transformations)

    def get_train_loader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return train_loader

    def get_val_loader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True, drop_last=False)
        return val_loader


# class niiDataset():
#     """CLEF dataset."""

#     def __init__(self, csv_file, root_dir, transform=None, train=True, noHU=False, reshape=(96, 256, 128), multiprocess=True, preload_path=None, save_fname="train.jl"):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.metadata = pd.read_csv(os.path.join(root_dir, csv_file), header=None)[24:36]
#         self.images = []
#         self.root_dir = root_dir
#         self.transform = transform
#         self.train = train
#         self.noHU = noHU
#         self.multiprocess = multiprocess
#         self.out = "/home/sentic/Documents/md/data/tuberculosis/out/"

#         self.right = True
#         self.crop_perct = (0, 0)
#         self.reshape = reshape

#         self.process_metadata()
#         if preload_path:
#             print("Loading...")
#             self.images = joblib.load(self.out + preload_path)
#         else:
#             self.load_all()
#             print(len(self.images))
#             joblib.dump(self.images, self.out + save_fname)
#         print("Loaded: " + str(self.__len__()))

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, idx):
#         return self.images[idx]

#     # def __getitem2__(self, idx):
#     #     if torch.is_tensor(idx):
#     #         idx = idx.tolist()

#     #     img_path = os.path.join(self.root_dir, self.metadata.iloc[idx, 0][:11])

#     #     # Read an image with OpenCV
#     #     import SimpleITK as sitk
#     #     image = sitk.ReadImage(img_path)

#     #     inimg_raw = sitk.GetArrayFromImage(image)
#     #     directions = np.asarray(image.GetDirection())
#     #     if len(directions) == 9:
#     #         inimg_raw = np.flip(inimg_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
#     #     del image  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#     #     if not self.noHU:
#     #         tvolslices, xnew_box = utils.preprocess(inimg_raw, resolution=[256, 256])
#     #         tvolslices[tvolslices > 600] = 600
#     #         tvolslices = np.divide((tvolslices + 1024), 1624)
#     #     else:
#     #         # support for non HU images. This is just a hack. The models were not trained with this in mind
#     #         tvolslices = skimage.color.rgb2gray(inimg_raw)
#     #         tvolslices = skimage.transform.resize(tvolslices, [256, 256])
#     #         tvolslices = np.asarray([tvolslices * x for x in np.linspace(0.3, 2, 20)])
#     #         tvolslices[tvolslices > 1] = 1
#     #         sanity = [(tvolslices[x] > 0.6).sum() > 25000 for x in range(len(tvolslices))]
#     #         tvolslices = tvolslices[sanity]

#     #     if self.train:
#     #         labels = self.metadata.iloc[idx, 1:]
#     #         labels = np.array(labels)
#     #         labels = labels.astype('float32').T

#     #         from scipy.ndimage import zoom

#     #         if self.right:
#     #             tvolslices = tvolslices[:, :, :128]
#     #         else:
#     #             tvolslices = tvolslices[:, :, 128:]

#     #         if self.crop_perct:
#     #             nslices = tvolslices.shape[0]
#     #             tvolslices = tvolslices[int(nslices * self.crop_perct[0]):nslices -
#     #                                     int(nslices * self.crop_perct[1]), ::]

#     #         # resize to same shape
#     #         trnsfs = (self.reshape[0] / tvolslices.shape[0], 1, 1)
#     #         tvolslices = zoom(tvolslices, trnsfs)

#     #         # check image
#     #         # print("writing: " + str(os.path.basename(img_path)))
#     #         # img_new = sitk.GetImageFromArray(tvolslices)
#     #         # sitk.WriteImage(img_new, "/home/sentic/Documents/md/data/tuberculosis/out/" + str(os.path.basename(img_path)) + ".nii.gz")
#     #         # return -1

#     #         return tvolslices[:, None, :, :], labels
#     #     else:
#     #         return basename(img_path), tvolslices

#     def process_image(self, i):
#         i = 2 * i
#         img_path = os.path.join(self.root_dir, self.metadata.iloc[i, 0])

#         # Read an image with OpenCV
#         import SimpleITK as sitk
#         image = sitk.ReadImage(img_path)

#         inimg_raw = sitk.GetArrayFromImage(image)
#         directions = np.asarray(image.GetDirection())
#         if len(directions) == 9:
#             inimg_raw = np.flip(inimg_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
#         del image
#         del directions

#         tvolslices, _ = utils.preprocess(inimg_raw, resolution=[256, 256])
#         tvolslices[tvolslices > 600] = 600
#         tvolslices = np.divide((tvolslices + 1024), 1624)
#         del inimg_raw

#         from scipy.ndimage import zoom
#         if self.crop_perct:
#             nslices = tvolslices.shape[0]
#             tvolslices = tvolslices[int(nslices * self.crop_perct[0]):nslices -
#                                     int(nslices * self.crop_perct[1]), ::]

#         # resize to same shape
#         trnsfs = (self.reshape[0] / tvolslices.shape[0], 1, 1)
#         tvolslices = zoom(tvolslices, trnsfs)

#         slices = []
#         for j in [0, 1]:
#             labels = self.metadata.iloc[i + j, 1:]
#             labels = np.array(labels)
#             labels = labels.astype('float32').T
#             if not j:  # right # reversed not !!!!!!!!!!!!!!!!!
#                 # print(str(j) + ": " + str(tvolslices.shape))
#                 slices.append((tvolslices[:, :, :128], labels))
#                 # print(str(j) + ": " + str(tvolslices.shape))
#             else:  # left
#                 # print(str(j) + ": " + str(tvolslices.shape))
#                 slices.append((tvolslices[:, :, 128:], labels))
#                 # print(str(j) + ": " + str(tvolslices.shape))

#         # return slices
#         # check image
#         print("writing: " + str(os.path.basename(img_path)))
#         for i, s in enumerate(slices):
#             img_new = sitk.GetImageFromArray(s[0])
#             labels_text = "_".join([str(x) for x in s[1]])
#             orient = i  # "_right_" # 'left' if 'left' in self.metadata.iloc[idx, 0] else 'right'
#             sitk.WriteImage(img_new, self.out + str(os.path.basename(img_path)) + "_" +
#                             str(orient) + "_" + labels_text + ".nii.gz")
#         return -1

#     def process_metadata(self):
#         # process metadata file
#         new = []
#         for i in range(self.metadata.shape[0]):
#             name = self.metadata.iloc[i, 0][:11]
#             labels = self.metadata.iloc[i, 1:].tolist()
#             orient = 0 if 'left' in self.metadata.iloc[i, 0] else 1
#             new.append([name, orient, *labels])
#         new = pd.DataFrame(new)
#         new = new.drop_duplicates()
#         # new = new.reset_index()
#         self.metadata = new
#         print(self.metadata)

#     def load_all(self):
#         from multiprocessing import Pool

#         print("Loading data...")
#         if self.multiprocess:
#             pool = Pool(8)                        # Create a multiprocessing Pool
#             # process data_inputs iterable with pool
#             self.images = list(tqdm(pool.imap(self.process_image, range(
#                 int(self.metadata.shape[0] / 2))), total=int(self.metadata.shape[0] / 2)))
#             pool.close()
#             pool.join()
#             self.images = [item for sublist in self.images for item in sublist]
#             print(len(self.images))
#             # pool.map(self.process_image, range(int(self.metadata.shape[0] / 2)))
#         else:
#             for i in tqdm(range(0, self.metadata.shape[0], 2)):
#                 process_image(i)


class niiDataset2():
    """CLEF dataset."""
    from smart_load import SmartCrop

    def __init__(self, csv_file, root_dir, out_dir=None, transform=None, train=True, noHU=False, reshape=(96, 256, 128), multiprocess=True, preload_path=None, save_fname="train.jl"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(os.path.join(root_dir, csv_file))#, header=None)
        self.images = []
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.noHU = noHU
        self.multiprocess = multiprocess
        self.out_dir = out_dir

        self.crop_perct = (0, 0)
        self.reshape = reshape
        self.min_max = (-600, 100)
        self.T1_WINDOW_LEVEL = (1500, -500)
        self.scrop = self.SmartCrop(output_size=(256, 512), maskthreshold=150, np_array=True)

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
        return self.images[idx]

    def to_image(self, volume, T1_WINDOW_LEVEL):
        img = sitk.GetImageFromArray(volume)
        img = sitk.Cast(sitk.IntensityWindowing(img,
                                                windowMinimum=T1_WINDOW_LEVEL[1] - T1_WINDOW_LEVEL[0] / 2.0,
                                                windowMaximum=T1_WINDOW_LEVEL[1] + T1_WINDOW_LEVEL[0] / 2.0),
                        sitk.sitkUInt8)
        img = sitk.GetArrayFromImage(img)
        return img

    def process_image(self, i):
        i = 2 * i
        img_path = os.path.join(self.root_dir, self.metadata.iloc[i, 0])

        # Read an image with OpenCV
        image = sitk.ReadImage(img_path)
        inimg_raw = sitk.GetArrayFromImage(image)
        directions = np.asarray(image.GetDirection())
        if len(directions) == 9:
            tvolslices = np.flip(inimg_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
        del image
        del directions

        # resize to depth
        trnsfs = (self.reshape[0] / tvolslices.shape[0], 1, 1)
        tvolslices = zoom(tvolslices, trnsfs)

        # smart crop
        new_slices = []
        for j in range(tvolslices.shape[0]):
            new_slices.append(self.scrop.call2(tvolslices[j, :, :], thresh=-300))
        tvolslices = np.asarray(new_slices)

        # divide left/right + to image
        slices = []
        for j in [0, 1]:
            labels = self.metadata.iloc[i + j, 1:]
            labels = np.array(labels)
            labels = labels.astype('float32').T
            if not j:  # right # reversed not !!!!!!!!!!!!!!!!!
                # print(str(j) + ": " + str(tvolslices.shape))
                img = self.to_image(tvolslices[:, :, :256], self.T1_WINDOW_LEVEL)
                for l in range(img.shape[0]):
                    slices.append((img[l,:, :], labels, basename(img_path) + "_left"))
                # print(str(j) + ": " + str(tvolslices.shape))
            else:  # left
                # print(str(j) + ": " + str(tvolslices.shape))
                img = self.to_image(tvolslices[:, :, 256:], self.T1_WINDOW_LEVEL)
                for l in range(img.shape[0]):
                    slices.append((img[l,:, :], labels, basename(img_path) + "_right"))
                # slices.append((tvolslices[:, :, 256:], labels, basename(img_path) + "_right"))
                # print(str(j) + ": " + str(tvolslices.shape))
        
        return slices



        # for j, (img, label, name) in enumerate(slices):
        #     img = Image.fromarray(img).convert('RGB')
        #     labels_text = "_".join([str(int(x)) for x in label])
        #     img.save(self.out_dir + name + "_" + "{:03d}".format(j) + "_" + labels_text + ".png", quality=1)
        # exit()

        # check image
        # print("writing: " + str(os.path.basename(img_path)))
        # for i, s in enumerate(slices):
        #     img_new = sitk.GetImageFromArray(s[0])
        #     labels_text = "_".join([str(x) for x in s[1]])
        #     orient = i #"_right_" # 'left' if 'left' in self.metadata.iloc[idx, 0] else 'right'
        #     sitk.WriteImage(img_new, self.out + str(os.path.basename(img_path)) + "_" +
        #                     str(orient) + "_" + labels_text + ".nii.gz")

        # for i, (volume, label, name) in enumerate(slices):
        #     l, w = (-500, 1500)
        #     min_max = [0, 0]
        #     min_max[0] = l - w // 2.0
        #     min_max[1] = l + w // 2.0
        #     # img = sitk.Image(volume)  # , sitk.sitkInt16
        #     # img = sitk.Cast(sitk.IntensityWindowing(img,
        #     #                                         windowMinimum=self.T1_WINDOW_LEVEL[1] - self.T1_WINDOW_LEVEL[0] / 2.0,
        #     #                                         windowMaximum=self.T1_WINDOW_LEVEL[1] + self.T1_WINDOW_LEVEL[0] / 2.0),
        #     #                 sitk.sitkUInt16)
        #     for j in range(volume.shape[0]):
        #         img = volume[j, :, :]

        #         img = sitk.GetImageFromArray(img)
        #         # image.SetSize((128, 256, 256))
        #         # img.CopyInformation(image)
        #         img = sitk.Cast(sitk.IntensityWindowing(img,
        #                                                 windowMinimum=self.T1_WINDOW_LEVEL[1] -
        #                                                 self.T1_WINDOW_LEVEL[0] / 2.0,
        #                                                 windowMaximum=self.T1_WINDOW_LEVEL[1] + self.T1_WINDOW_LEVEL[0] / 2.0),
        #                         sitk.sitkUInt8)
        #         img = sitk.GetArrayFromImage(img)
        #         # images = []
        #         # for l, w in ((-500, 1500),):  # ((-500, 600), (-40, 600), (40, 400)):
        #         #     min_max = [0, 0]
        #         #     min_max[0] = l - w // 2.0
        #         #     min_max[1] = l + w // 2.0

        #         #     # img[img > self.min_max[1]] = self.min_max[1]
        #         #     # img[img < self.min_max[0]] = self.min_max[0]
        #         #     # # img = np.asarray(np.divide((img - self.min_max[0]), self.min_max[1]-self.min_max[0])*255)
        #         #     # img = (img - min(img)) / (max(img) - min(img)) * 255

        #         #     # image = sitk.ReadImage(fdata("nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT1.nrrd"))
        #         #     # img = sitk.Image(img, sitk.sitkInt16)

        #         #     # sitk.Cast(sitk.IntensityWindowing(img,
        #         #     #                            windowMinimum=min_max[0],
        #         #     #                            windowMaximum=min_max[1]),
        #         #     #                            sitk.sitkUInt16)
        #         #     images.append(img)
        #         # if len(images) == 1:
        #         #     images = images[0]
        #         #     images = np.asarray(images)  # .astype('uint8')
        #         # else:
        #         #     images = np.asarray(images)  # .astype('uint8')
        #         #     images = np.rollaxis(images, 0, 3)

        #         img = Image.fromarray(img).convert('RGB')
        #         labels_text = "_".join([str(int(x)) for x in label])
        #         img.save(self.out_dir + name + "_" + "{:03d}".format(j) + "_" + labels_text + ".png", quality=1)
        # exit()

    def process_metadata(self):
        # process metadata file
        new = []
        for i in range(self.metadata.shape[0]):
            name = self.metadata.iloc[i, 0][:15]
            labels = self.metadata.iloc[i, 1:].tolist()
            orient = 0 if 'left' in self.metadata.iloc[i, 0] else 1
            new.append([name, orient, *labels])
        new = pd.DataFrame(new)
        new = new.drop_duplicates()
        # new = new.reset_index()
        self.metadata = new
        print(self.metadata)

    def load_all(self):
        from multiprocessing import Pool

        print("Loading data...")
        if self.multiprocess:
            pool = Pool(16)                        # Create a multiprocessing Pool
            # process data_inputs iterable with pool
            self.images = list(tqdm(pool.imap(self.process_image, range(
                int(self.metadata.shape[0] / 2))), total=int(self.metadata.shape[0] / 2)))
            pool.close()
            pool.join()
            self.images = [item for sublist in self.images for item in sublist]
            print(len(self.images))
            # pool.map(self.process_image, range(int(self.metadata.shape[0] / 2)))
        else:
            for i in tqdm(range(0, self.metadata.shape[0], 2)):
                self.process_image(i)

class sfDataset():
    def __init__(self, data_dir, transform):
        self.data = None
        self.data_dir = data_dir
        self.transform = transform
        self.parse_dir()

    def parse_dir(self):
        import glob
        files = glob.glob(self.data_dir + "/*/*")
        files = [x for x in files if "flipped" not in x]
        # CTR_TRN_007_right_093.png
        file_details = []
        for file in files:
            path = file
            label, fname = file.split("/")[-2], file.split("/")[-1]
            # fname = basename(file)
            # name, _, location, _ = fname.split(".")
            # _, side, slicen = location.split("_")

            name = fname[:len("TRN_0000")]
            side, slicen = fname[:-4].split("_")[-2], fname[:-4].split("_")[-1]

            file_details.append((name, side, int(slicen), label, path))
        file_details = pd.DataFrame(file_details)
        file_details.columns = ["name", "side", "slicen", "label", "path"]
        groups = file_details.groupby(["name", "side", "slicen"])
        self.data = list(groups)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_group = self.data[idx][1]
        image_path = image_group.iloc[0]["path"]
        image = Image.open(image_path)

        labels = image_group["label"].tolist()

        affected = 0
        caverns = 0
        pleurisy = 0

        if "caverns" in labels:
            caverns = 1
        if "pleurisy" in labels:
            pleurisy = 1
        if caverns or pleurisy or "affected" in labels:
            affected = 1


        labels = np.asarray([affected, caverns, pleurisy])
        labels = labels.astype('float').T
        # sample = {'image': image, 'label': labels}
        if self.transform:
            sample = self.transform(image)

        return image, labels
    

class pngDatasetMinivol():
    """CLEF dataset."""
    from smart_load import SmartCrop

    def __init__(self, csv_file, root_dir, out_dir=None, transform=None, train=True, noHU=False, reshape=(96, 256, 128), multiprocess=True, preload_path=None, save_fname="train.jl"):
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
        self.train = train
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
        tvolslices = []
        for pic in pics:
            pic = cv2.imread(pic)
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            tvolslices.append(pic)
        tvolslices = np.stack(tvolslices)
    
        # resize to depth
        # trnsfs = (self.reshape[0] / tvolslices.shape[0], 1, 1)
        # tvolslices = zoom(tvolslices, trnsfs)

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

class niiDatasetMinivol():
    """CLEF dataset."""
    from smart_load import SmartCrop

    def __init__(self, csv_file, root_dir, out_dir=None, transform=None, train=True, noHU=False, reshape=(96, 256, 128), multiprocess=True, preload_path=None, save_fname="train.jl"):
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
        self.train = train
        self.noHU = noHU
        self.multiprocess = multiprocess
        self.out_dir = out_dir

        self.crop_perct = (0, 0)
        self.reshape = reshape
        self.min_max = (2500, -1000)
        self.T1_WINDOW_LEVEL = (2500, -1000)
        self.scrop = self.SmartCrop(output_size=(256, 512), maskthreshold=150, np_array=True)

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

    def to_image(self, volume, T1_WINDOW_LEVEL):
        img = sitk.GetImageFromArray(volume)
        img = sitk.Cast(sitk.IntensityWindowing(img,
                                                windowMinimum=T1_WINDOW_LEVEL[1] - T1_WINDOW_LEVEL[0] / 2.0,
                                                windowMaximum=T1_WINDOW_LEVEL[1] + T1_WINDOW_LEVEL[0] / 2.0),
                        sitk.sitkUInt8)
        img = sitk.GetArrayFromImage(img)
        return img

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
        image = sitk.ReadImage(img_path)
        inimg_raw = sitk.GetArrayFromImage(image)
        directions = np.asarray(image.GetDirection())
        if len(directions) == 9:
            tvolslices = np.flip(inimg_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
        del image
        del directions

        # resize to depth
        trnsfs = (self.reshape[0] / tvolslices.shape[0], 1, 1)
        tvolslices = zoom(tvolslices, trnsfs)

        # smart crop
        new_slices = []
        for j in range(tvolslices.shape[0]):
            new_slices.append(self.scrop.call2(tvolslices[j, :, :], thresh=-1600))
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


class NiiDatasetMinivol_new():
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

    def process_metadata(self):
        # process metadata file
        new = []
        for i in range(self.metadata.shape[0]):
            name = self.metadata.iloc[i, 0][:15]
            labels = self.metadata.iloc[i, 1:].tolist()
            orient = 0 if 'left' in self.metadata.iloc[i, 0] else 1
            new.append([name, orient, *labels])
        new = pd.DataFrame(new)
        new = new.drop_duplicates()
        # new = new.reset_index()
        self.metadata = new
        print(self.metadata)

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


class createNiiDataset():
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
        self.T1_WINDOW_LEVEL = (1500, -500)
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
            new_slices.append(self.scrop.call2(tvolslices[j, :, :], thresh=-300))
        tvolslices = np.asarray(new_slices)

        tvolslices = self.to_image(tvolslices[:, :, :], self.T1_WINDOW_LEVEL)
        for i, row in group.iterrows():
            slicen = row["slicen"]
            side = row["side"]
            label = row["label"]

            img = tvolslices[slicen, :, :]
            shape = tvolslices.shape
            img_prev = tvolslices[slicen+2, :, :] if slicen+2<shape[0] else tvolslices[slicen, :, :]
            img_next = tvolslices[slicen-2, :, :] if slicen-2>=0 else tvolslices[slicen, :, :]

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
        self.T1_WINDOW_LEVEL = [(1500, -500), (350, 40), (500, -600)]
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
            new_slices.append(self.scrop.call2(tvolslices[j, :, :], thresh=-300))
        tvolslices = np.asarray(new_slices)

        tvolslices1 = self.to_image(tvolslices[:, :, :], self.T1_WINDOW_LEVEL[0])
        tvolslices2 = self.to_image(tvolslices[:, :, :], self.T1_WINDOW_LEVEL[1])
        tvolslices3 = self.to_image(tvolslices[:, :, :], self.T1_WINDOW_LEVEL[2])

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

class niiDataLoader():
    """
    CLEF data loading class using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, num_workers=12):
        self.transformations = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = niiDataset("train.csv", data_dir, self.transformations)
        self.val_dataset = niiDataset("val.csv", data_dir, self.transformations)

    def get_train_loader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return train_loader

    def get_val_loader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True, drop_last=False)
        return val_loader


if __name__ == '__main__':

    dataloader = CLEFDataLoader("/home/sentic/Documents/md/ImageCLEF2019/data/clef_projections_v1.1/", 32, 4)
    train_loader = dataloader.get_train_loader()
    val_loader = dataloader.get_val_loader()

    for x, y in train_loader:
        print(x.shape)
        print(type(x))
        print(y.shape)
        print(type(x))
        break
