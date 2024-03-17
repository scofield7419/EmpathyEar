import os
from openai import OpenAI
import numpy as np
import time
import random
import torch
from sentence_transformers import SentenceTransformer, util

random.seed(13)

data_train_dialog = np.load("/mnt/haofei/MSA/MERG/data/empathetic-dialogue-emb/sys_dialog_texts.train.npy", allow_pickle=True)
data_train_target = np.load("/mnt/haofei/MSA/MERG/data/empathetic-dialogue-emb/sys_target_texts.train.npy", allow_pickle=True)

assert len(data_train_dialog) == len(data_train_target)
datalen = len(data_train_dialog)
print(datalen)

id = 0
conv_data, tmp = [], []
while id < datalen:
    utter = data_train_dialog[id]
    utter.append(data_train_target[id])
    conv_data.append(utter)
    id = id + 1

print(len(conv_data))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('/mnt/haofei/MSA/Data/HuggingFace/all-mpnet-base-v2').to(device)

traindata = []
for conv in conv_data:
    tmp = ""
    for u in conv:
        tmp += u.strip() + " "
    traindata.append(tmp)

traindata_embedding = model.encode(traindata, convert_to_tensor=True, device=device)

def get_maxsim(querydata, k=5):
    testdata_embedding = model.encode(querydata, convert_to_tensor=True, device=device)
    cosine_scores = util.cos_sim(testdata_embedding, traindata_embedding)[0].detach().cpu().numpy()
    sort = np.argsort(cosine_scores)[::-1]
    fewshot_indices, fewshot_sim_scores = sort[:k], [cosine_scores[idx] for idx in sort[:k]]
    # print(fewshot_indices)

    return fewshot_indices


def main():
    save_path = '/mnt/haofei/MSA/MERG/data/empathetic-dialogue-emb/sensim_5_ED_train.txt'
    data_test = np.load("/mnt/haofei/MSA/MERG/data/empathetic-dialogue-emb/sys_dialog_texts.train.npy", allow_pickle=True)
    len_test = len(data_test)
    # sensims = open("results/sensim_5_ED.txt", mode="r").readlines()

    sensim_list = []
    continueid = -1     # continue after interruption, default to -1
    id = continueid + 1
    while id < len_test:
        testdata = []
        for d in data_test[id]:
            tmp = ""
            for j in d:
                tmp += j.strip() + " "
            testdata.append(tmp)

        sensims = get_maxsim(testdata, 5)
        sensim_list.append(sensims)
        print(str(id)+'\n')
        id += 1
    with open(save_path, "w") as f:
        for item in sensim_list:
            f.write(str(item) + "\n")

if __name__ == "__main__":
    main()