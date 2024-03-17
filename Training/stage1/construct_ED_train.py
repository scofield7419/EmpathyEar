### EP

import os
import json
from openai import OpenAI
import numpy as np
import time
import random
import torch
from sentence_transformers import SentenceTransformer, util


random.seed(13)

### initial prompt

##### zero-shot
context = "Speaker: I’ve been hearing some strange noises around the house at night.\n\
    Listener: oh no! That’s scary! What do you think it is?\n\
Speaker: I don’t know, that’s what’s making me anxious.\n\
    Listener:"

prompt_EP_0 = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener.\n\n" + "The following is the existing dialogue context:\n\n" + context

##### few-shot
prompt_EP_1 = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener.\n\n" + "The following is an instance:\n\n"

prompt_EP_5 = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener.\n\n" + "The following is some instances:\n\n"


data_train_dialog = np.load("/mnt/haofei/MSA/MERG/data/empathetic-dialogue-emb/sys_dialog_texts.train.npy", allow_pickle=True)
data_train_target = np.load("/mnt/haofei/MSA/MERG/data/empathetic-dialogue-emb/sys_target_texts.train.npy", allow_pickle=True)

assert len(data_train_dialog) == len(data_train_target)

datalen = len(data_train_dialog)
print(datalen)
shot1 = random.randint(0, datalen-1)
shot5 = random.sample(range(0, datalen), 5)
shot = 1

instance = ''
for _ in range(shot):
    for i, text in enumerate(data_train_dialog[shot1]):
        if i % 2 == 0:
            utt = 'Speaker:' + text + '\n'
            instance += utt
        else:
            utt = 'Listener:' + text + '\n'
            instance += utt
    instance += 'Empathetic Response:'
    instance += data_train_target[shot1]

save_path = '/mnt/haofei/MSA/MERG/data/empathetic-dialogue-emb/ED_train_1shot.json'
id = 0
dialogue = []
while id < datalen:
    dia = ''
    for i, text in enumerate(data_train_dialog[id]):
        if i % 2 == 0:
            utt = 'Speaker:' + text + '\n'
            dia += utt
        else:
            utt = 'Listener:' + text + '\n'
            dia += utt
    context = prompt_EP_1 + instance + 'Following is the dialogue context:\n' + dia + 'Empathetic Response:'
    target = data_train_target[id]
    dialogue.append({'context': context, 'target': target})
    id = id + 1
print(len(dialogue))
json_data = json.dumps(dialogue)
with open(save_path, "w") as f:
    f.write(json_data)

    ### save as reuse, for saving time
    # fw = open("./sensim_5.txt", mode="w")
    # dialog_test = np.load('data/ED/sys_dialog_texts.test.npy', allow_pickle=True)
    # testdata = []
    # for d in dialog_test:
    #     tmp = ""
    #     for j in d:
    #         tmp += j.strip() + " "
    #     testdata.append(tmp)

    # id = 0
    # len_test = len(testdata)
    # k = 5

    # while id < len_test:
    #     testdata_embedding = model.encode(testdata[id], convert_to_tensor=True, device=device)
    #     cosine_scores = util.cos_sim(testdata_embedding, traindata_embedding)[0].detach().cpu().numpy()
    #     sort = np.argsort(cosine_scores)[::-1]
    #     fewshot_indices, fewshot_sim_scores = sort[:k], [cosine_scores[idx] for idx in sort[:k]]
    #     fw.write(str(fewshot_indices) + '\n')
    #     id = id + 1

    # fw.close()