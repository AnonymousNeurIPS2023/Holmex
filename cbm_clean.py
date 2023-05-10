from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy
import pickle
from util import *

# options -------------------------------------------------------------------------------------------------------------
# set random seed
seeds = [1234, 2345, 3456, 4567, 5678]

task_str = "7"
class_num = 5
epochs_num = 2000
# clip_model_str ='ViT-B/32'
# store_path = 'CLIP_ViT_B_32'
clip_model_str ='ViT-L/14'
store_path = 'CLIP_ViT_L_14'
batch_size = 100

# ----------------------------------------------------------------------------------------------------------------------
log_path = "tasks/task"+task_str+"/"+store_path+"/cbm_clean_log.txt"
log_file = open(log_path, 'w')
cbm_clean_acc_list = []

train_path = "tasks/task"+task_str+"/"+store_path+"/data_train.pickle"
print(train_path)
f = open(train_path, 'rb')
X, y = pickle.load(f)
f.close()
train_data = MyData(X=torch.tensor(X, dtype=torch.float), y=torch.tensor(y, dtype=torch.int64))

f = open("tasks/task"+task_str+"/"+store_path+"/data_test.pickle", 'rb')
X, y = pickle.load(f)
f.close()
test_data = MyData(X=torch.tensor(X, dtype=torch.float), y=torch.tensor(y, dtype=torch.int64))

for seed_num in range(len(seeds)):
    seed = seeds[seed_num]
    setup_seed(seed)
    print(seed)

    dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    f = open("tasks/task"+task_str+"/label_list.pickle", 'rb')
    label_list = pickle.load(f)
    f.close()
    addition_concepts = []
    for word in addition_concepts:
        if word not in label_list:
            label_list.append(word)

    print(label_list)
    cbm_clean = CBM(concept_list=label_list, class_num=class_num, clip_model_str=clip_model_str)
    optimizer_cbm = Adam(params=cbm_clean.parameters(), weight_decay=0.001)

    # training -------------------------------------------------------------------------------------------------------------
    for epochs in range(epochs_num):
        loss = 0
        for batch_X, batch_y in dataloader:
            batch_y = torch.reshape(batch_y, (len(batch_y), ))

            optimizer_cbm.zero_grad()
            loss = cross_entropy(cbm_clean(batch_X), batch_y)
            loss.backward()
            optimizer_cbm.step()

        if epochs % 100 == 99:
            print("cbm")
            train_acc = evaluation(cbm_clean, train_data)
            print(train_acc)
            test_acc = evaluation(cbm_clean, test_data)
            print(test_acc)

    with torch.no_grad():
        data = test_data
        print("final:")
        print("cbm")
        cbm_clean_acc = evaluation(cbm_clean, test_data)
        print(cbm_clean_acc)
        print(cbm_clean.linear.weight)

    # store result
    f = open("tasks/task"+task_str+"/"+store_path+"/cbm_clean.pickle"+str(seed), 'wb')
    pickle.dump(cbm_clean, f)
    f.close()

    # log --------------------------------------------------------------------------------------------------------------
    log_file.write("words_list: " + str(label_list) + '\n')
    log_file.write("seed: " + str(seed) + '\n')
    log_file.write("epochs_num: " + str(epochs_num) + '\n')
    log_file.write("batch_size: " + str(batch_size) + '\n')
    log_file.write("cbm_clean_acc: " + str(cbm_clean_acc) + '\n')
    cbm_clean_acc_list.append(cbm_clean_acc)
log_file.close()

print("cbm_clean_acc_list")
for i in range(len(cbm_clean_acc_list)):
    print(cbm_clean_acc_list[i])
