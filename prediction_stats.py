import joblib
from numpy.core.fromnumeric import argmax
import pandas as pd
from glob import glob
from os.path import basename
from smart_load import SmartCrop

root_data_path = "/home/sentic/Documents/data/imageclef2021/"
dataset_path = root_data_path + "affections_copy/"
out_path = root_data_path + "out/"

def aggregate_dataset(dataset_path):
    data = []
    
    # ex CTR_TRN_013.nii.gz_right_031.png
    def parse_filename(name):
        s = name.split("_")
        no = s[2].split(".")[0]
        side = s[3]
        slice = s[4].split(".")[0]
        return no, side, slice

    label_paths = glob(dataset_path + "/*")
    for label_path in label_paths:
        label = basename(label_path)
        file_paths = glob(label_path + "/*")
        for file_path in file_paths:
            if 'flipped' in file_path:
                continue
            name = basename(file_path)
            no, side, slice = parse_filename(name)
            data.append([name, no, side, slice, label])
    
    data = pd.DataFrame(data)
    data.columns = ["name", "no", "side", "slice", "label"]
    return data

from sklearn.metrics import precision_score
def compute_stats():
    # labels = pd.read_excel(labels_path)
    # preds = pd.read_excel(preds_path)
    pass

def link_type_tb(data1, data2):
    data1["FileName"] = ["TRN_0" + x[8:11] + ".nii.gz"  for x in data1["name"].to_list()]
    data = data1.merge(data2, how='left', on="FileName")
    return data

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import SimpleITK as sitk



class createNiiDataset2():
    """CLEF dataset."""
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

        self.scrop = SmartCrop(output_size=(256, 512), maskthreshold=-500, np_array=True)

        # internal
        self.min_max = (2500, -1000)
        self.T1_WINDOW_LEVEL = [(1500, -500), (350, 40), (500, -600)]

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
        name = group.iloc[0]["FileName"]
        img_path = os.path.join(self.root_dir, name)

        # Read an image with OpenCV
        image = sitk.ReadImage(img_path)
        inimg_raw = sitk.GetArrayFromImage(image)
        directions = np.asarray(image.GetDirection())
        if len(directions) == 9:
            tvolslices = np.flip(inimg_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
        del image
        del directions


        # # smart crop
        new_slices = []
        for j in range(tvolslices.shape[0]):
            new_slices.append(self.scrop.call2(tvolslices[j, :, :], thresh=-1600)) # -1200
        tvolslices = np.asarray(new_slices)
        tvolslices = self.to_image(tvolslices[:, :, :], self.min_max)

        def win_scale(data, wl, ww, dtype, out_range):
            """
            Scale pixel intensity data using specified window level, width, and intensity range.
            """
            
            data_new = np.empty(data.shape, dtype=np.double)
            data_new.fill(out_range[1]-1)
            
            data_new[data <= (wl-ww/2.0)] = out_range[0]
            data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = \
                ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
            data_new[data > (wl+ww/2.0)] = out_range[1]-1
            
            return data_new.astype(dtype)

        # tvolslices[tvolslices < -3000] = -3000
        # tvolslices[tvolslices > 2000] = 2000

        # tvolslices1 = win_scale(tvolslices, -750, 1000, np.uint8, (-2000, 2000))
        # tvolslices1 = self.to_image(tvolslices[:, :, :], self.min_max)
        # tvolslices2 = self.to_image(tvolslices[:, :, :], self.T1_WINDOW_LEVEL[1])
        # tvolslices3 = self.to_image(tvolslices[:, :, :], self.T1_WINDOW_LEVEL[2])

        for i, row in group.iterrows():
            slicen = row["slice"]
            side = row["side"]
            label = row["TypeOfTB"] #TypeOfTB

            img = tvolslices[slicen, :, :]
            img_prev = tvolslices[slicen, :, :]
            img_next = tvolslices[slicen, :, :]

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

            label_dir_path = self.out_dir + str(label)
            if not os.path.exists(label_dir_path):
                os.makedirs(label_dir_path)
            img.save(label_dir_path + "/" + name + "_" + side + "_" + str(slicen) + ".png")


    def generate_train(self):
        from multiprocessing import Pool

        groups = self.data.groupby("FileName")
        groups = [x[1] for x in groups][:-1] # oneee is missing !!!!!!!!!!!!!!!!!!!

        print("Loading data...")
        print(int(len(groups)))
        if self.multiprocess:
            pool = Pool(4)                        # Create a multiprocessing Pool
            # process data_inputs iterable with pool
            self.images = list(tqdm(pool.imap(self.process_image, groups), total=int(len(groups))))
            pool.close()
            pool.join()
        else:
            for group in tqdm(groups):
                self.process_image(group)


from os import mkdir
from os.path import isdir
import nibabel as nib
# def extract_slices(data, data_path):
#     groups = data.groupby('FileNme')
#     for group in groups:
#         vol = nib.load(data_path + group["FileName"].to_list()[0]).get_fdata()
#         for row in group.iterrows():
#             slice = vol[row["slice"]]


import nibabel as nib

def show_vol_info():
    nii_root_path = "/home/sentic/Documents/data/imageclef2021/data/"
    out_path = "/home/sentic/Documents/data/imageclef2021/"

    nii_paths = glob(nii_root_path + "*.nii.gz")

    info = []
    for nii_path in sorted(nii_paths):
        vol = nib.load(nii_path)
        header = vol.header
        info.append([basename(nii_path), header])
    info = pd.DataFrame(info)
    info.columns = ["name", "header"]
    info.to_excel(out_path+"info.xlsx")

def list_duplicates(seq):
  seen = set()
  seen_add = seen.add
  # adds all elements it doesn't know yet to seen and all other to seen_twice
  seen_twice = set( x for x in seq if x in seen or seen_add(x) )
  # turn the set into a list (as requested)
  return list( seen_twice )

def process_results(results_path, prefix="0"):
    results = pd.read_csv(results_path)
    results = print(results["2"].to_list()[0])
    results.columns = ["indecsi", "name", "1", "2", "3", "4", "5", "6"]
    #----------------
    print(results.shape)
    results = results.loc[~results["name"].str.contains("temp")]
    print(results.shape)
    #----------------

    # results = results[["name", "1", "2", "3", "4", "5", "6"]]
    print(results)
    results["root_name"] = [basename(x).split(".")[0] for x in results["name"].to_list()]
    results["slicen"] = [int(x.split("_")[-1].split(".")[0]) for x in results["name"].to_list()]
    results["side"] = ["left" if "left" in x else "right" for x in results["name"].to_list()]
    labels_dict = dict()

    groups = results.groupby("root_name")
    for group_name, group in groups:
        group_name = prefix + "_" + group_name
        labels_dict[group_name] = {"left":"", "right":""}
        group_sides = group.groupby("side")
        for side_name, group_side in group_sides:
            group_side = group_side.sort_values(by="slicen")
            if group_side.shape[0]!=128:
                print("ALEEERT!!!")
                print(group_side.shape)
                print(group_name)
                print(side_name)
                print(group_side["name"].to_list())
                print(list(set(group_side["name"].to_list())))

                return
            labels = np.array(group_side.loc[:, ["1", "2", "3", "4", "5", "6"]])
            labels_dict[group_name][side_name] = labels
    return labels_dict

from numpy import mean
def process_results2(results_path, prefix="0"):
    results = pd.read_csv(results_path)
    print(results)
    results = results.loc[results["0"].str.contains("covid")]
    results = results.loc[~results["0"].str.contains("temp")]
    results.columns = ["indecsi", "name", "1", "2", "3", "4", "5", "6"]
    results = results.drop_duplicates(subset=["name"])
    results = results[["name", "1", "2", "3"]]
    print(results)

    print(results["name"].to_list()[0])
    results["root_name"] = ["_".join(basename(x).split("_")[:-2]) for x in results["name"].to_list()]
    results["slicen"] = [int(x.split("_")[-1].split(".")[0]) for x in results["name"].to_list()]
    results["side"] = ["left" if "left" in x else "right" for x in results["name"].to_list()]
    results["label"] = ["non-covid" if "non-covid" in x else "covid" for x in results["name"].to_list()]

    def get_missing(results):
        slicen = results["slicen"].to_list()
        for i in range(94):
            if not slicen[i] + 1 == slicen[i+1]:
                print(results["name"].to_list()[0] + "----------------: " + str(i+1))
                # print(float(results.loc[results["slicen"] == i]["1"]))
                new_row = [results["name"].to_list()[0].replace("_0.png", "_" + str(i+1) + ".png"),\
                      mean([float(results.loc[results["slicen"] == i]["1"]), float(results.loc[results["slicen"] == i+2]["1"])]),\
                      mean([float(results.loc[results["slicen"] == i]["2"]), float(results.loc[results["slicen"] == i+2]["2"])]),\
                      mean([float(results.loc[results["slicen"] == i]["3"]), float(results.loc[results["slicen"] == i+2]["3"])]),\
                    #   mean(results[i, "2"], results[i+2, "2"]),\
                      results["root_name"].to_list()[0], i+1, results["side"].to_list()[0], results["label"].to_list()[0] ]
                # group_side.insert
                results.loc[-1] = new_row

    labels_dict = dict()

    groups = results.groupby(["root_name", "label"])
    for group_name, group in groups:
        group_name = prefix + "_" + group_name[0] + "_" + group_name[1]
        labels_dict[group_name] = {"left":"", "right":""}
        group_sides = group.groupby("side")
        for side_name, group_side in group_sides:
            group_side = group_side.sort_values(by="slicen")
            get_missing(group_side)
            group_side = group_side.sort_values(by="slicen")

            if group_side.shape[0]!=96:
                print(group_side["slicen"].to_list())
                print(group_side)
                print("ALEEERT!!!")
                print(group_side.shape)
                print(group_name)
                print(side_name)
                print(group_side["name"].to_list())
                print(list(set(group_side["name"].to_list())))

                return
            labels = np.array(group_side.loc[:, ["1", "2", "3"]])
            labels_dict[group_name][side_name] = labels
    return labels_dict

def train_results(results, train_type="log1"):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split

    X, y = [], []
    for x in results:
        # print(x)
        # print(len(results[x]["left"]))
        # print("----------------")
        # print(results[x]["right"])
        # print("----------------")
        # exit()
        point = np.ndarray.flatten(np.concatenate([results[x]["left"], results[x]["right"]], axis=0))
        
        # print(point.shape)
        # print(name)

        try:
            y.append(1 if "non-covid" in x else 0)
            X.append(point)
        except:
            print("Skipped " + name)

    # print(len(X))
    # print(np.stack(X).shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)
    
    from sklearn.linear_model import LogisticRegression
    if train_type == "mlp1":
        clf = MLPClassifier(max_iter=300, warm_start=True, activation='relu', solver='adam', hidden_layer_sizes=(100)).fit(X_train, y_train)
    elif train_type == "mlp2":
        clf = MLPClassifier(max_iter=40, alpha=1e-5, hidden_layer_sizes=(100, 30)).fit(X_train, y_train)
    elif train_type == "log":
        clf = LogisticRegression(max_iter=300, warm_start=True).fit(X, y) # 0.8908296943231441

    clf.predict_proba(X_test[:1])
    clf.predict(X_test[:5])

    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))

    return clf


from sklearn.model_selection import KFold
def train_results_folds(results, train_type="log1"):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split

    X, y = [], []
    for x in results:
        # print(x)
        # print(len(results[x]["left"]))
        # print("----------------")
        # print(results[x]["right"])
        # print("----------------")
        # exit()
        point = np.ndarray.flatten(np.concatenate([results[x]["left"], results[x]["right"]], axis=0))
        
        # print(point.shape)
        # print(name)

        try:
            y.append(1 if "non-covid" in x else 0)
            X.append(point)
        except:
            print("--------------------------------------------Skipped " + name)
    X = np.array(X)
    y = np.array(y)

    # print(len(X))
    # print(np.stack(X).shape)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)
    
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    clfs = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        from sklearn.linear_model import LogisticRegression
        if train_type == "mlp1":
            clf = MLPClassifier(max_iter=300, warm_start=True, activation='relu', solver='adam', hidden_layer_sizes=(100)).fit(X_train, y_train)
        elif train_type == "mlp2":
            clf = MLPClassifier(max_iter=40, alpha=1e-5, hidden_layer_sizes=(100, 30)).fit(X_train, y_train)
        elif train_type == "log":
            clf = LogisticRegression(max_iter=300, warm_start=True).fit(X, y) # 0.8908296943231441

        # if i%3==0:
        #     clf = MLPClassifier(max_iter=300, warm_start=True, activation='relu', solver='adam', hidden_layer_sizes=(100)).fit(X_train, y_train)
        # elif i%3==1:
        #     clf = MLPClassifier(max_iter=40, alpha=1e-5, hidden_layer_sizes=(100, 30)).fit(X_train, y_train)
        # elif i%3==2:
        #     clf = LogisticRegression(max_iter=300, warm_start=True).fit(X, y) # 0.8908296943231441

        print(clf.score(X_train, y_train))
        print(clf.score(X_test, y_test))
        clfs.append(clf)

    return clfs

def merge_datasets():
    root_path = "/home/sentic/Documents/data/imageclef2021/"
    data2020 = root_path + "info_2020.xlsx"
    data2021 = root_path + "info_2021.xlsx"

    data2020 = pd.read_excel(data2020, engine='openpyxl')
    # remove duplicates that are out of range >300
    data2021 = pd.read_excel(data2021, engine='openpyxl').iloc[:300]

    dups2020 = list_duplicates(data2020["header"].to_list())
    dups2021 = list_duplicates(data2021["header"].to_list())

    print(data2020.loc[data2020["header"].isin(dups2020)])
    print(data2021.loc[data2021["header"].isin(dups2021)])

    data2020 = data2020.loc[~data2020["header"].isin(dups2020)]
    data2021 = data2021.loc[~data2021["header"].isin(dups2021)]

    print(data2020.shape)
    print(len(data2020.header.unique()))
    print(data2021.shape)
    print(len(data2021.header.unique()))

    data = data2020.merge(data2021, how="inner", on="header")
    data.to_excel(root_path + "merger.xlsx")

    print(data2020.shape)
    print(data.shape)

def parse_manual_annotations(path):
    data = open(path, "rt").read().strip().split("\n\n")
    
    slices = []
    for section in data:
        lines = section.split("\n")
        label = lines[0].strip()
        print(label)
        for line in lines[1:]:
            print(line)
            line = line.split(",")
            print(line)
            no, side, r = line[0].split()
            slices.append([label, no, side, r])
            for ranges in line[1:]:
                slices.append([label, no] + ranges.strip().split())
    expanded = []
    for slice in slices:
        # print(slice)
        r = [int(x) for x in slice[3].split("-")]
        for i in range(r[0], r[1]+1):
            expanded.append(slice[:-2] + ["left" if slice[-2]=="r" else "right", i])
    return expanded

from sklearn.model_selection import train_test_split
from shutil import copy
def split_train_test(data_path, out_path):
    # get files and labels
    files, labels = [], []
    label_folders = glob(data_path + "*")
    for label_folder in label_folders:
        label_files = glob(label_folder + "/*")
        files += label_files
        labels += [basename(label_folder)] *len(label_files)

    # split
    X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.1, random_state=42, stratify=labels)
    
    # write data
    for x, y in zip(X_train, y_train):
        out = out_path + "train_set/" + str(y)
        if not os.path.exists(out):
            os.mkdir(out)
        copy(x, out + "/" + basename(x))

    for x, y in zip(X_test, y_test):
        out = out_path + "test_set/" + str(y)
        if not os.path.exists(out):
            os.mkdir(out)
        copy(x, out + "/" + basename(x))

if __name__ == "__main__":
    # data = aggregate_dataset(dataset_path)
    # data.to_excel(out_path)
    # print(data)

    # radu_labels_path = root_data_path + "labels.xlsx"
    # radu_labels2_path = root_data_path + "labels2.xlsx"

    # merger_path = root_data_path + "merger.xlsx"
    # vol_path = root_data_path + "data/"
    # train_path = root_data_path + "train2/"

    # clef_labels_path = root_data_path + "cleaned_metaData.csv"
    # preds_path = root_data_path + "out/checkpoint_full_train_efficientnet_albs.pth.tar_aggregate.xlsx"

    # data1 = pd.read_excel(radu_labels_path, engine='openpyxl')
    # data2 = pd.read_csv(clef_labels_path)

    # split data
    # split_train_test("/data/train_minivol/", "/data/")
    # exit()

    # make test files
    # test_data_dir = root_data_path + "test/"
    # test_files = sorted(glob(test_data_dir + "*"))
    # metaData_file_path = root_data_path + "test_metaData.csv"

    # test_list = [["FileName", "TypeOfTB"]]
    # for test_file in test_files:
    #     name = basename(test_file)
    #     test_list.append([name.split(".")[0] + "_left." +".".join(name.split(".")[1:]), 0])
    #     test_list.append([name.split(".")[0] + "_right." +".".join(name.split(".")[1:]), 0])
        
    # test_df = pd.DataFrame(test_list)
    # test_df.to_csv(metaData_file_path, header=None, index=None)
    # exit()

    # process results
    # results_path = "/home/sentic/Documents/sentic/Cosmin/slice_based/res_train.csv"
    # result1 = process_results(results_path, prefix="1")

    # results_path = "/home/sentic/storage2/iccv/out/checkpoint_2_full_minivol_albs_ce.pth.tar_iccv_results.jl.csv"
    # result2 = process_results(results_path, prefix="")
    # result2.to_csv("/home/sentic/storage3/sentic/Cosmin/slice_based/processed_results.csv")
    # exit()

    # --------------------------------

    # val_result.to_csv("/home/sentic/storage3/sentic/Cosmin/slice_based/processed_val_results.csv")
    
    # from scipy.ndimage import zoom

    # trnsfs = (self.reshape[0] / , 1, 1)
    # tvolslices = zoom(tvolslices, trnsfs)

    out_path = "/home/sentic/storage3/sentic/Cosmin/slice_based/"
    train_results_path = "/home/sentic/storage3/sentic/Cosmin/slice_based/res_train_iccv2.csv"
    val_results_path = "/home/sentic/storage3/sentic/Cosmin/slice_based/res_val_iccv.csv"
    test_results_path = "/home/sentic/storage3/sentic/Cosmin/slice_based/res_test_iccv.csv"
    train_result = process_results2(train_results_path, prefix="")
    joblib.dump(train_result, out_path + basename(train_results_path) + "_dict.jl")
    print("-------processed train------------")
    val_result = process_results2(val_results_path, prefix="")
    joblib.dump(val_result, out_path + basename(val_results_path) + "_dict.jl")
    print("-------processed val------------")
    test_result = process_results2(test_results_path, prefix="")
    joblib.dump(test_result, out_path + basename(test_results_path) + "_dict.jl")
    # print("-------processed test------------")


# ----------------------

    def predict_folds(clfs, X):
        preds = []
        for clf in clfs:
            preds.append(clf.predict_proba(X))
        preds = np.asarray(preds)
        preds = np.mean(preds, axis=0)
        y_pred = (preds[:,1] >= 0.5).astype(bool)
        return y_pred

# --------------------------------------------
    result = joblib.load(out_path + basename(train_results_path) + "_dict.jl")
    # clf = train_results(result, train_type="log")
    clfs = train_results_folds(result, train_type="log")

    X, y = [], []
    for x in val_result:
        point = np.ndarray.flatten(np.concatenate([val_result[x]["left"], val_result[x]["right"]], axis=0))

        try:
            y.append(1 if "non-covid" in x else 0)
            X.append(point)
        except:
            print("Skipped " + x)

    from sklearn.metrics import f1_score as f1

    
    # y_pred = clf.predict(X)
    y_pred = predict_folds(clfs, X)
    print(y_pred)
    # y_pred = (clf.predict_proba(X)[:,1] >= 0.99).astype(bool)
    print(f1(y, y_pred))



#---------------------------
    X, names = [], []
    for x in test_result:
        point = np.ndarray.flatten(np.concatenate([test_result[x]["left"], test_result[x]["right"]], axis=0))
        name = x 

        X.append(point)
        names.append(name.split("_")[1])
        # print(X[0].shape)
        # print(X[0].shape)
        for i, x in enumerate(X):
            if x.shape!=(576,):
                print("!!!!")
                print(names[i])
                print(X[i].shape)
                print(test_result[names[i].split(".")[0]]["left"].shape)
                print(test_result[names[i].split(".")[0]]["right"].shape)
                print(test_result[names[i].split(".")[0]]["right"])
                # copy(test_path + names[i] + "_right_128.png", out_path)

    X = np.stack(X)
    # preds = clf.predict(X)
    # probs = clf.predict_proba(X)

    preds = predict_folds(clfs, X)

    covid = []
    non_covid = []
    for pred, name in zip(preds, names):
        if pred == 0:
            covid.append(name)
        else:
            non_covid.append(name)
    open(out_path + "covid.csv", "w").write(",".join(covid))
    open(out_path + "non_covid.csv", "w").write(",".join(non_covid))


    # joblib.dump(names, out_path + "names.jl")
    # joblib.dump(probs, out_path + "probs.jl")
    # pd.DataFrame(zip(names, preds)).to_csv(out_path + "results.csv", index=False, header=None)
    exit()
# --------------------------------------------


    for train_type in ["log", "mlp1", "mlp2"]:
        result = joblib.load(out_path + basename(val_results_path) + "_dict.jl")
        clf = train_results(result, train_type=train_type)
        for res in [test_results_path]:
            result_test = process_results2(test_results_path)
            X, names = [], []
            for x in result_test:
                point = np.ndarray.flatten(np.concatenate([result_test[x]["left"], result_test[x]["right"]], axis=0))
                name = x 

                X.append(point)
                names.append(name)
            
            test_path = root_data_path + "out/test_set/1/"
            out_path = root_data_path + "out/"

            print(X[0].shape)
            for i, x in enumerate(X):
                if x.shape!=(1536,):
                    print("!!!!")
                    print(names[i])
                    print(X[i].shape)
                    print(result_test[names[i].split(".")[0]]["left"].shape)
                    print(result_test[names[i].split(".")[0]]["right"].shape)
                    print(result_test[names[i].split(".")[0]]["right"])
                    # copy(test_path + names[i] + "_right_128.png", out_path)

            X = np.stack(X)
            preds = clf.predict(X)
            probs = clf.predict_proba(X)

            joblib.dump(names, out_path + "names.jl")
            joblib.dump(probs, out_path + "probs2singrev_" + train_type + "_" + res + ".jl")
            pd.DataFrame(zip(names, preds)).to_csv(out_path + "results.csv", index=False, header=None)
    # exit()


    exit()
    # --------------------------------
    # results_path = "/home/sentic/Documents/sentic/Cosmin/slice_based/res_train_full.csv"
    # result3 = process_results(results_path, prefix="3")

    # results_path = "/home/sentic/Documents/sentic/Cosmin/slice_based/res_train_full_flipped.csv"
    # result4 = process_results(results_path, prefix="4")

    # result = dict(result1, **result2)
    # result.update(result3)
    # result.update(result4)
    # result = result2

    # joblib.dump(result, out_path + "result_dict.jl")
    # result = joblib.load(out_path + "result_dict.jl")

    # clf = train_results(result)
    # exit()

    # parse manual annotations
    # manual_annotations_path = root_data_path + "manual_annotations.txt"
    # data = parse_manual_annotations(manual_annotations_path)
    # data = pd.DataFrame(data)
    # data.columns = ["TypeOfTB", "no", "side", "slice"]
    # data["FileName"] = ["TRN_" + str(i).zfill(4) + ".nii.gz" for i in data["no"].to_list()]
    # print(data)
    # data.to_csv(root_data_path + "annotations.csv")
    # train_path = root_data_path + "train3/"
    # # createNiiDataset_minivol(data, vol_path, train_path).generate_train()
    # exit()

    # extract dubious slices
    # preds = pd.read_csv("/home/sentic/Documents/sentic/Cosmin/slice_based/res_train.csv")
    # limit = 0.6
    # preds = preds.loc[(preds["1"] < limit) & (preds["2"] < limit) & (preds["3"] < limit) & (preds["4"] < limit) & (preds["5"] < limit) & (preds["6"] < limit)]
    # print(preds)
    # print(preds.shape)
    # data["0"] = ["/data/out/entire_train/1/" + n + "_" + side + "_" + str(s) + ".png" for n, side, s in zip(data["FileName"].to_list(), data["side"].to_list(), data["slice"].to_list())]
    # preds = preds.loc[~preds["0"].isin(data["0"].to_list())]
    # preds["FileName"] = ["_".join(basename(x).split("_")[:2]) for x in preds["0"]]
    # print(preds.shape)

    # labels = pd.read_csv(clef_labels_path)
    # preds = preds.merge(labels, on="FileName")
    # print(preds)

    # corrections_path = out_path + "corrections/"
    # from pathlib import Path
    # for name, row in preds.iterrows():
    #     Path(corrections_path + str(row["TypeOfTB"])).mkdir(parents=True, exist_ok=True)
    #     print(corrections_path + str(row["TypeOfTB"]))
    #     exit()
    #     copy("/home/sentic/Documents/data/imageclef2021" + row["0"][5:], corrections_path + str(row["TypeOfTB"]))
    # exit()

    # test process model

    # for train_type in ["log", "mlp1", "mlp2"]:
    #     result = joblib.load(out_path + "result_dict.jl")
    #     clf = train_results(result, train_type=train_type)
    #     for res in ["res.csv", "res_full.csv", "res_flipped.csv", "res_full_flipped.csv"]:
    #         results_path = "/home/sentic/Documents/sentic/Cosmin/slice_based/" + res
    #         result_test = process_results(results_path)
    #         X, names = [], []
    #         for x in result_test:
    #             point = np.ndarray.flatten(np.concatenate([result_test[x]["left"], result_test[x]["right"]], axis=0))
    #             name = x + ".nii.gz"

    #             X.append(point)
    #             names.append(name)
            
    #         test_path = root_data_path + "out/test_set/1/"
    #         out_path = root_data_path + "out/"
    #         for i, x in enumerate(X):
    #             if x.shape!=(1536,):
    #                 print("!!!!")
    #                 print(names[i])
    #                 print(X[i].shape)
    #                 print(result_test[names[i].split(".")[0]]["left"].shape)
    #                 print(result_test[names[i].split(".")[0]]["right"].shape)
    #                 print(result_test[names[i].split(".")[0]]["right"])
    #                 copy(test_path + names[i] + "_right_128.png", out_path)

    #         X = np.stack(X)
    #         preds = clf.predict(X)
    #         probs = clf.predict_proba(X)

    #         joblib.dump(names, out_path + "names.jl")
    #         joblib.dump(probs, out_path + "probs2singrev_" + train_type + "_" + res + ".jl")
    #         # pd.DataFrame(zip(names, preds)).to_csv(out_path + "results.csv", index=False, header=None)
    # exit()

    # # compile results
    # names = [x[2:] for x in joblib.load(out_path + "names.jl")]
    # probs = glob(out_path + "probs*.jl")
    # print(probs)
    # data = []
    # for prob in probs:
    #     data.append(joblib.load(prob))

    # """2 eficient-net models trained on slices -> softmax features on normal and flipped images -> loglin + mlp models -> mean of probs -> labels"""
    # print(len(data))
    # mean_prob = sum(data)/len(data)
    # labels = argmax(mean_prob, axis=1)+1
    # print(labels)
    # pd.DataFrame(zip(names, labels)).to_csv(out_path + "results.csv", index=False, header=None)
    # exit()

    # data = link_type_tb(data1, data2)

    # # data.to_excel(root_data_path + "labels2.xlsx", engine='openpyxl')
    # data["TypeOfTB"] = [y if x == 'ok' else 6 for x, y in zip(data["label"].to_list(), data["TypeOfTB"].to_list())]
    # print(data["TypeOfTB"].value_counts(normalize=True))

    data = pd.read_excel(radu_labels_path, engine='openpyxl')
    merger = pd.read_excel(merger_path, engine='openpyxl')

    # # merge and convert names
    # name_dict = {x:y for x, y in zip(merger["name_x"].to_list(), merger["name_y"].to_list())}
    # data["name"] = [x[:18] for x in data["name"].to_list()]
    # data = data.loc[data["name"].isin(merger["name_x"].to_list())]
    # print(data.shape)
    # data["FileName"] = [name_dict[x] for x in data["name"].to_list()]

    # data = data.merge(data2, how='left', on="FileName")

    # data["TypeOfTB"] = [y if x != 'ok' else 6 for x, y in zip(data["label"].to_list(), data["TypeOfTB"].to_list())]
    # print(data["TypeOfTB"].value_counts(normalize=True))

    # createNiiDataset2(data, vol_path, train_path).generate_train()

    # parse manual annotations
    manual_annotations_path = root_data_path + "manual_annotations.txt"
    data = parse_manual_annotations(manual_annotations_path)
    data = pd.DataFrame(data)
    data.columns = ["TypeOfTB", "no", "side", "slice"]
    data["FileName"] = ["TRN_" + str(i).zfill(4) + ".nii.gz" for i in data["no"].to_list()]
    print(data)
    data.to_csv = root_data_path + "annotations.csv"
    train_path = root_data_path + "train3/"
    createNiiDataset2(data, vol_path, train_path).generate_train()


    # show_vol_info()
    # merge_datasets()


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # what is happening with missing predictions??