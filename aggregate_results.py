from posixpath import basename
import joblib
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

csv_file = "val.csv"
data_dir = "/home/sentic/Documents/md/data/tuberculosis/images/"
out_dir = "/home/sentic/Documents/md/data/tuberculosis/out/slices/"
data_path = "/home/sentic/Documents/data/Madu/Medical/LUNGS/EXPERIMENTS/data/"

csv_file = "/data/test_metaData.csv" # labels csv
data_dir = "/data/out/" # /home/sentic/storage3/sentic/Cosmin/slice_based/ #root images path
out_dir = "/data/out/" #"/home/sentic/storage2/iccv/out/"
data_path = "/home/sentic/Documents/md/data/tuberculosis/images/" # not used

label2int = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e':4, 'f':5}

# label2int = {'affected': 0, 'caverns': 1, 'ok': 2, 'pleurisy': 3}

# checkpoint =   "checkpoint_full_train_efficientnet_albs.pth.tar" #checkpoint_full_train_efficientnet_albs    #"checkpoint_albs.pth.tar" # "model_best.pth.tar" #"model_best_albs.pth.tar" #"model_best_albs2.pth.tar" #"model_best_check.pth.tar" #"model_best_resnet50_1gpu_albs.pth.tar" #"./checkpoint_resnet50_1gpu.pth.tar"
# results_fname = checkpoint + "_val_results0.jl"
# out_file = out_dir + checkpoint + "_aggregate.xlsx"

# print(out_file)

# # df1 = pd.read_csv(out_dir + "CTRfnet_minivol_sigmoid.csv", header=None)
# # df2 = pd.read_csv(out_dir + "CTRfnet_minivol.csv", header=None)
# # df3 = pd.read_csv(out_dir + "CTRfnet.csv", header=None)

# # # print(np.asarray(df1[1].tolist()).reshape(-1, 1))

# # df = np.asarray(df1[1].tolist()).reshape(-1, 1)+np.asarray( df2[2].tolist()).reshape(-1, 1) + np.asarray(df3[3].tolist()).reshape(-1, 1)
# # df = df/3
# # print(df)
# # exit()


# results = joblib.load(out_dir + results_fname)
# print(results[0])


maxs_results = []

def process_results(results, batch_size=128):
    softmax = nn.Softmax(dim=1)

    scores = [x[0] for x in results]
    labels = [x[1] for x in results]
    names = np.asarray([x[2] for x in results])

    scores = torch.cat(scores, 0)
    size = scores[0].shape
    scores = torch.reshape(scores, (-1, batch_size, *size))

    labels = torch.cat(labels, 0)
    size = labels[0].shape
    labels = torch.reshape(labels, (-1, batch_size, *size))

    names = np.reshape(names, (-1, batch_size))

    print("labels: " + str(labels.shape))
    print("scores: " + str(scores.shape))
    print("names: " + str(names.shape))

    results = []
    for score, label, name in zip(scores, labels, names):
        score = np.max(softmax(score).cpu().numpy(), 0)
        label = label[0].cpu().numpy()
        name = name[0]
        results.append((score, label, name))
    return results


def process_results_flat(results):
    softmax = nn.Softmax(dim=1)
    print(results)
    scores = [x[0] for x in results]
    labels = [x[1] for x in results]
    names = np.asarray([x[2] for x in results])

    scores = torch.cat(scores, 0)
    size = scores[0].shape
    scores = torch.reshape(scores, (-1, *size)).cpu().numpy()

    labels = torch.cat(labels, 0)
    size = labels[0].shape
    labels = torch.reshape(labels, (-1, *size)).cpu().numpy()

    names = np.reshape(names, (-1, 1))
    # names = names[:, np.newaxis]

    # print(names.shape)
    # print(scores.shape)
    # print(labels.shape)
    # labels = labels[:, np.newaxis]
    all = np.concatenate((names, scores, labels), axis=1)
    print(all.shape)

    print("labels: " + str(labels.shape))
    print("scores: " + str(scores.shape))
    print("names: " + str(names.shape))

    data = pd.DataFrame(all)
    # data.to_excel(out_file)
    return data

def process_results_iccv(results):
    # print(len(results))
    # print(len(results[0]))
    # print(len(results[1]))
    # print(len(results[2]))

    # print(results[0])
    # print(len(results[0][2]))


    scores = [x[0] for x in results]
    labels = [x[1] for x in results]
    names = np.asarray([x[2] for x in results])

    names_all = []
    for name in names:
        names_all.extend(name)
    names = np.asarray(names_all)
    
    # print(names.shape)
    # print(names[0])

    scores = torch.cat(scores, 0)
    size = scores[0].shape
    scores = torch.reshape(scores, (-1, *size)).cpu().numpy()
    scores = scores[:, :6]

    labels = torch.cat(labels, 0)
    size = labels[0].shape
    labels = torch.reshape(labels, (-1, *size)).cpu().numpy()

    names = np.reshape(names, (-1, 1))
    labels = labels[:, np.newaxis]

    print("labels: " + str(labels.shape))
    print("scores: " + str(scores.shape))
    print("names: " + str(names.shape))

    all = np.concatenate((names, scores, labels), axis=1)
    print(all.shape)



    data = pd.DataFrame(all)
    # data.to_excel(out_file)
    return data
    

# label2int = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e':4, 'f':5}
checkpoint = "checkpoint_2_minivol_albs.pth.tar" #checkpoint_full_train_efficientnet_albs    #"checkpoint_albs.pth.tar" # "model_best.pth.tar" #"model_best_albs.pth.tar" #"model_best_albs2.pth.tar" #"model_best_check.pth.tar" #"model_best_resnet50_1gpu_albs.pth.tar" #"./checkpoint_resnet50_1gpu.pth.tar"
out_file = out_dir + checkpoint + "_iccv_aggregate.xlsx"
# full_data = pd.DataFrame()
# # checkpoint_2_full_minivol_albs_ce.pth.tar_iccv_results.jl.csv
# # for i in range(1):
# results_fname = checkpoint + "_iccv_results.jl"
# results = joblib.load(out_dir + results_fname)
# # results = pd.read_csv(out_dir + results_fname)
# data = process_results_iccv(results)
# # full_data = pd.concat([full_data, data])
# # data = pd.DataFrame(data)
# full_data = data
# full_data.to_excel(out_file)
# exit()
data = pd.read_excel(out_file, engine='openpyxl')
data = data.reset_index()
data = data[[0, 1, 2, 3, 4, 5, 6, 7]]

from glob import glob
def filter_results(data, low_thr=0.4, high_thr=0.99):
    # print(data)
    # print(data[6])
    data = data.drop_duplicates(subset=[0])
    data = data.loc[~data[0].str.contains("dummy")]
    data = data.loc[(np.array([float(x) for x in data[6]])>high_thr) | (np.array([float(x) for x in data[6]])<low_thr)]
    data["healthy"] = [1 if float(x)>high_thr else 0 for x in data[6]]
    # print(data.shape)
    # print(len(data[0].to_list()))

    for x in data[0].to_list():
        if not len(basename(x).split("_")) > 2:
            print(x)

    data["no"] = [basename(x).split("_")[2] for x in data[0].to_list()]
    data["sliceno"] = [float(basename(x).split("_")[4].split(".")[0]) for x in data[0].to_list()]

    # data["name"] = [if basename(x).split("_")[2] for x in data["0"].to_list()]

    def get_nos(path="/data/train/"):
        folder_list = []
        folders = glob(path + "*/*")
        for folder in folders:
            folder_list.append((0 if "/covid" in folder else 1, basename(folder).split("_")[2], float(len(glob(folder + "/*")))))
        folders = pd.DataFrame(folder_list)
        folders.columns = [7, "no", "nos"]
        return folders

    folders = get_nos()
    merger = data.merge(folders, on=[7, "no"])
    merger.to_excel(out_file + "filtered.xlsx")
    merger = merger.loc[(merger["sliceno"]<0.7*merger["nos"]) | (merger["sliceno"]>0.3*merger["nos"])]
    print(merger)
    return merger

data = filter_results(data)

data = pd.read_excel(out_file + "filtered.xlsx", engine='openpyxl')
data = data.loc[(data["sliceno"]<0.7*data["nos"]) & (data["sliceno"]>0.3*data["nos"])]
print(data)
data = data.sample(frac=1, random_state=42)
from shutil import copy
out_folder = "/data/out/filter/"
def create_dataset(data):
    for i, row in data.iterrows():
        # print(row)
        # label = row[0].split("/")[-2]
        # name = "_".join(basename(row[0]).split("_")[:3])
        if row["healthy"] == 1:
            copy(row[0], out_folder + "healthy/")
        else:
            if row[7] == 0:
                copy(row[0], out_folder + "covid/")
            else:
                copy(row[0], out_folder + "non_covid/")

# data = data.iloc[:2000]d
create_dataset(data)
exit()

# df1 = pd.read_csv(out_dir + "CTRfnet_minivol_sigmoid.csv", header=None)
# df2 = pd.read_csv(out_dir + "CTRfnet_minivol.csv", header=None)
# df3 = pd.read_csv(out_dir + "CTRfnet.csv", header=None)

# # print(np.asarray(df1[1].tolist()).reshape(-1, 1))

# df = np.asarray(df1[1].tolist()).reshape(-1, 1)+np.asarray( df2[2].tolist()).reshape(-1, 1) + np.asarray(df3[3].tolist()).reshape(-1, 1)
# df = df/3
# print(df)
# exit()


# results = joblib.load(out_dir + results_fname)
# print(results[0])

# process_results_flat()
# exit()

# for i in range(0, len(results), 2):
#     scores1, labels1, names1 = results[i]
#     scores2, labels2, names2 = results[i+1]

#     scores = torch.cat((scores1, scores2), 0)
#     print(scores.shape)
#     maxs = np.max(softmax(scores).cpu().numpy(), 0)

#     labels1 = labels1.cpu().numpy()
#     labels2 = labels2.cpu().numpy()

#     maxs_results.append((list(maxs), list(labels1[0]), names1[0]))

# results = process_results(results)
# outputs = np.asarray([[x[0][0], x[0][1], x[0][3]] for x in results])
# targets = np.asarray([x[1] for x in results])
# names = [x[2] for x in results]

#------------------------------
results = process_results(full_data)
outputs = np.asarray([x[0] for x in results])
targets = np.asarray([x[1] for x in results])
names = [x[2] for x in results]

#----------------------------
import pandas as pd

df = pd.DataFrame()

df["names"] = np.asarray(names)[0::2]
# df2["rnames"] = np.asarray(names)[1::2]

df["affected"] = outputs[::2, 0]
df["raffected"] = outputs[1::2, 0]

# df["caverns"] = outputs[::2, 1]
# df["rcaverns"] = outputs[1::2, 1]

# df["pleurisy"] = outputs[::2, 2]
# df["rpleurisy"] = outputs[1::2, 2]


print(df)
df["names"] = [x[:-5] + ".nii.gz" for x in df["names"].tolist()]
df.to_csv(out_dir + "CTRfnet_minivol_sigmoid.csv", header=None, index=None)
exit()
#----------------------------

def load():
    pass

def auc(output, target):
    """Computes the auc for each column"""
    nclasses = output.shape[1]
    aucs = []
    for i in range(nclasses):
        try:
            aucs.append(roc_auc_score(target[:, i:i + 1], output[:, i:i + 1]))
        except ValueError:
            aucs.append(0)
    mean_auc = np.mean(aucs)
    aucs.append(mean_auc)
    return aucs

print(auc(outputs, targets))

# "./checkpoint_resnet50_1gpu.pth.tar"
# [0.9099190283400809, 0.9409603778535817, 0.9649621212121212, 0.9386138424685946]

# "model_best_resnet50_1gpu_albs.pth.tar"
# [0.9251012145748988, 0.940697979532931, 0.959280303030303, 0.941693165712711]

# "model_best_resnet50_1gpu_albs.pth.tar"
# [0.7827260458839406, 0.8076620309630018, 0.8229166666666666, 0.8044349145045363]

# "model_best_albs2.pth.tar"
# [0.9261133603238866, 0.9527683022828655, 0.9526515151515151, 0.943844392586089]

# "model_best_albs.pth.tar"
# [0.939608636977058, 0.9517187090002625, 0.959280303030303, 0.9502025496692079]

# "model_best.pth.tar"
# [0.9257759784075573, 0.9393859879296773, 0.9848484848484849, 0.9500034837285732]