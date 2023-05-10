import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy
import pickle
from util import *

# set random seed
# setup_seed(1234)
# store_path = 'CLIP_ViT_L_14'
store_path = 'CLIP_ViT_B_32'
words_list_embedding_str = 'words_list_embedding'
# words_list_embedding_str = 'words_list_embedding_PCBM'
target_str = 'bed'
spurious_str = 'dog'
task_str = "8"


train_path = "tasks/task" + task_str + "/" + store_path + "/data_train.pickle"
print(train_path)
f = open(train_path, 'rb')
X, y = pickle.load(f)
train_data = MyData(X=torch.tensor(X, dtype=torch.float), y=torch.tensor(y, dtype=torch.int64))

spurious_num_list = []
for experiment_num in range(100):
    print("experiment_num: ", experiment_num)
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # train full Linear ------------------------------------------------------------------------------------------------
    f = open("tasks/task"+task_str+"/label_list.pickle", 'rb')
    label_list = pickle.load(f)
    label_num = len(label_list)

    full = LinearModel(input_dim=len(X[0]), output_dim=len(label_list))
    optimizer = Adam(params=full.parameters(), weight_decay=0.01)

    for epochs in range(50):
        loss = 0
        for batch_X, batch_y in dataloader:
            batch_y = torch.reshape(batch_y, (len(batch_y), ))

            optimizer.zero_grad()
            loss = cross_entropy(full(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        if epochs % 10 == 9:
            print("full")
            train_acc = evaluation(full, train_data)
            print(train_acc)

    # detecting --------------------------------------------------------------------------------------------------------


    def get_key(item):
        return item[-1]


    with torch.no_grad():
        f = open("tasks/task" + task_str + "/label_list.pickle", 'rb')
        label_list = pickle.load(f)
        label_num = len(label_list)

        concept_source = "concepts_source/" + store_path + "/"+words_list_embedding_str+".pickle"
        print(concept_source)
        f = open(concept_source, 'rb')
        concept_label_and_embedding = pickle.load(f)
        weight = full.linear.weight

        record_list = list()
        for i in range(len(concept_label_and_embedding)):
            item = list()
            # append concept label
            item.append(concept_label_and_embedding[i][0])
            # get concept's vector
            concept_embedding = concept_label_and_embedding[i][1]

            score_list = list()
            # max - average
            for j in range(label_num):
                score_list.append(float(torch.dot(concept_embedding, weight[j])))
            max_index = score_list.index(max(score_list))
            item.append(label_list[max_index])
            effect_score = max(score_list) - sum(score_list) / len(score_list)
            item.append(effect_score)

            record_list.append(item)
        record_list.sort(key=get_key, reverse=True)
        num = 0
        spurious_num = 1953
        for i in range(len(record_list)):
            # print(i, ": ", record_list[i])
            if record_list[i][1] == target_str:
                # print(i, ": ", record_list[i])
                num += 1
                if record_list[i][0] == spurious_str:
                    spurious_num = num
        print(spurious_num)
        spurious_num_list.append(spurious_num)

print()
for i in range(len(spurious_num_list)):
    print(spurious_num_list[i])
