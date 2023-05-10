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
epochs_num = 1000
hidden_dim = 50
# store_path = 'CLIP_ViT_B_32'
store_path = 'CLIP_ViT_L_14'
# store_path = 'resnet50'

# ----------------------------------------------------------------------------------------------------------------------
log_path = "tasks/task" + task_str + "/" + store_path + "/mlp_log.txt "
log_file = open(log_path, 'w')
train_path = "tasks/task"+task_str+"/"+store_path+"/data_train.pickle"
print(train_path)
f = open(train_path, 'rb')
X, y = pickle.load(f)
train_data = MyData(X=torch.tensor(X, dtype=torch.float), y=torch.tensor(y, dtype=torch.int64))

f = open("tasks/task"+task_str+"/"+store_path+"/data_test.pickle", 'rb')
X, y = pickle.load(f)
test_data = MyData(X=torch.tensor(X, dtype=torch.float), y=torch.tensor(y, dtype=torch.int64))

for seed_num in range(len(seeds)):
    seed = seeds[seed_num]
    setup_seed(seed)
    print(seed)
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # train MLP --------------------------------------------------------------------------------------------------------
    model = MLPModel(input_dim=len(X[0]), hidden_dim=hidden_dim, output_dim=class_num)
    optimizer = Adam(params=model.parameters(), weight_decay=0.01)


    for epochs in range(epochs_num):
        loss = 0
        for batch_X, batch_y in dataloader:
            batch_y = torch.reshape(batch_y, (len(batch_y), ))

            optimizer.zero_grad()
            loss = cross_entropy(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        if epochs % 100 == 99:
            print("MLP")
            train_acc = evaluation(model, train_data)
            print(train_acc)
            test_acc = evaluation(model, test_data)
            print(test_acc)


    with torch.no_grad():
        data = test_data
        print("final:")
        print("MLP")
        acc = evaluation(model, test_data)
        print(acc)

    # store model
    f = open("tasks/task"+task_str+"/"+store_path+"/mlp.pickle"+str(seed), 'wb')
    pickle.dump(model, f)

    # log --------------------------------------------------------------------------------------------------------------
    log_file.write("seed: " + str(seed) + '\n')
    log_file.write("epochs_num: " + str(epochs_num) + '\n')
    log_file.write("hidden_dim: " + str(hidden_dim) + '\n')
    log_file.write("batch_size: " + str(batch_size) + '\n')
    log_file.write("acc: " + str(acc) + '\n')

log_file.close()
