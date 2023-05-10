import torch
from torch import nn
from torch.utils.data import DataLoader
import clip
import numpy as np
import random
import copy
from PIL import Image
import torchvision.transforms as transforms

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)

    def forward(self, x):
        output = self.linear(x)
        return output


# class CBM_sigmoid(nn.Module):
#     def __init__(self, concept_list, class_num):
#         super().__init__()
#         self.b = torch.randn(len(concept_list))
#         self.concept_list = concept_list
#         concept_list.append("other")
#         clip_model, preprocess = clip.load("ViT-B/32")
#         self.label_embeddings = clip_model.encode_text(clip.tokenize(concept_list))
#         # normalize
#         for i in range(len(concept_list)):
#             self.label_embeddings[i] = self.label_embeddings[i] / torch.norm(self.label_embeddings[i])
#         # compare to other
#         for i in range(len(concept_list)-1):
#             self.label_embeddings[i] = self.label_embeddings[i] - self.label_embeddings[-1]
#
#         self.label_embeddings = self.label_embeddings.detach().numpy()
#         self.concept = torch.asarray(self.label_embeddings[0:len(concept_list)-1], requires_grad=True, dtype=torch.float)
#
#         self.linear = nn.Linear(in_features=len(concept_list)-1, out_features=class_num, bias=False)
#
#     def forward(self, x):
#         x = torch.matmul(x, self.concept.t())
#         x = torch.sigmoid(x + self.b)
#         output = self.linear(x)
#         return output
class CBM_in_PCBM(nn.Module):
    def __init__(self, concept_list, class_num, clip_model_str="ViT-B/32"):
        super().__init__()
        self.concept_list = concept_list
        concept_list.append("other")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load(clip_model_str, device=device)
        self.label_embeddings = clip_model.encode_text(clip.tokenize(concept_list))
        # normalize
        for i in range(len(concept_list)):
            self.label_embeddings[i] = self.label_embeddings[i] / torch.norm(self.label_embeddings[i])
        # # compare to other
        # for i in range(len(concept_list)-1):
        #     self.label_embeddings[i] = self.label_embeddings[i] - self.label_embeddings[-1]

        self.label_embeddings = self.label_embeddings.detach().numpy()
        self.concept = torch.asarray(self.label_embeddings[0:len(concept_list)-1], requires_grad=True, dtype=torch.float)

        self.linear = nn.Linear(in_features=len(concept_list)-1, out_features=class_num, bias=False)

    def forward(self, x):
        x = torch.matmul(x, self.concept.t())
        output = self.linear(x)
        return output


class CBM(nn.Module):
    def __init__(self, concept_list, class_num, clip_model_str="ViT-B/32"):
        super().__init__()
        self.concept_list = concept_list
        concept_list.append("other")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load(clip_model_str, device=device)
        self.label_embeddings = clip_model.encode_text(clip.tokenize(concept_list))
        # normalize
        for i in range(len(concept_list)):
            self.label_embeddings[i] = self.label_embeddings[i] / torch.norm(self.label_embeddings[i])
        # compare to other
        for i in range(len(concept_list)-1):
            self.label_embeddings[i] = self.label_embeddings[i] - self.label_embeddings[-1]

        self.label_embeddings = self.label_embeddings.detach().numpy()
        self.concept = torch.asarray(self.label_embeddings[0:len(concept_list)-1], requires_grad=True, dtype=torch.float)

        self.linear = nn.Linear(in_features=len(concept_list)-1, out_features=class_num, bias=False)

    def forward(self, x):
        x = torch.matmul(x, self.concept.t())
        output = self.linear(x)
        return output


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        output = self.mlp(x)
        return output


class LinearMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, x):
        output = self.mlp(x)
        return output


class MyData(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)


class CelebAData(torch.utils.data.Dataset):
    def __init__(self, files_paths, image_lists):
        # files_paths should contain the file path of each class
        # image_lists is a list like [class_one_list, class_two_list, ...]
        self.files_paths = files_paths
        self.image_lists = image_lists
        self.class_num = np.zeros(len(image_lists), dtype=int)
        for i in range(len(image_lists)):
            self.class_num[i] = len(image_lists[i])
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        class_index = 0
        while index >= self.class_num[class_index]:
            class_index += 1
            index = index - self.class_num[class_index]
        y = torch.LongTensor(1)
        y[0] = class_index
        path = self.files_paths[class_index]+"/"+self.image_lists[class_index][index]
        image = Image.open(path)
        image = self.transform(image)
        X = image
        return X, y[0]

    def __len__(self):
        return sum([len(self.image_lists[i]) for i in range(len(self.image_lists))])


def evaluation(model, data:MyData):
    with torch.no_grad():
        count = 0
        for i in range(len(data)):
            input = torch.reshape(data.X[i], (1, len(data.X[i])))
            pred = model(input)
            output = torch.argmax(pred)
            if output == data.y[i]:
                count+=1
        return count/len(data)


def evaluation_CelebA(model, data:MyData):
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=True)
    with torch.no_grad():
        count = 0
        acc = 0
        for batch_X, batch_y in dataloader:
            print(batch_X.shape)
            output = model(batch_X)
            pred = torch.argmax(output, dim=1)
            acc += sum(pred == batch_y).item()
            count += 1
            if count % 1 == 0:
                print(count)
                print(acc)
        return acc/len(data)


def transfer_evaluation(black_model, white_model, white_model_edit, data_black:MyData, data_white:MyData,
                        weight=(0.5, 1, -0.5)):
    # print(weight)
    count = 0
    for i in range(len(data_black)):
        input_black = torch.reshape(data_black.X[i], (1, len(data_black.X[i])))
        input_white = torch.reshape(data_white.X[i], (1, len(data_white.X[i])))
        # Original
        pred_original = black_model(input_black)
        # before cbm
        pre_before = white_model(input_white)
        # after
        pre_after = white_model_edit(input_white)
        # final
        pred = pred_original * weight[0] + pre_after * weight[1] + pre_before * weight[2]
        output = torch.argmax(pred)
        if output == data_black.y[i]:
            count += 1
    return count / len(data_black)


def ensemble_evaluation(model1, model2, data1:MyData, data2:MyData):
    count = 0
    for i in range(len(data1)):
        input1 = torch.reshape(data1.X[i], (1, len(data1.X[i])))
        input2 = torch.reshape(data2.X[i], (1, len(data2.X[i])))
        # model1
        pred_1 = model1(input1)
        # model2
        pre_2 = model2(input2)
        # final
        pred = (pred_1+pre_2)/2
        output = torch.argmax(pred)
        if output == data1.y[i]:
            count += 1
    return count / len(data1)


def get_concept_embedding(concept_list, clip_model_str="ViT-B/32", compare_world="other"):
    str_list = copy.deepcopy(concept_list)
    if compare_world != None:
        str_list.append("compare_world")

    clip_model, preprocess = clip.load(clip_model_str)
    embeddings = clip_model.encode_text(clip.tokenize(str_list))
    # normalize
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i] / torch.norm(embeddings[i])

    if compare_world != None:
        # compare to other
        for i in range(len(str_list) - 1):
            embeddings[i] = embeddings[i] - embeddings[-1]
        embeddings = embeddings[0:len(concept_list)]
    return embeddings


# def assemble_evaluation2(model1, model2, data:MyData):
#     count = 0
#     for i in range(len(data)):
#         input = torch.reshape(data.X[i], (1, len(data.X[i])))
#         # model1
#         pred_1 = model1(input)
#         pred_1 = torch.softmax(pred_1, dim=1)
#         # model2
#         pre_2 = model2(input)
#         pre_2 = torch.softmax(pre_2, dim=1)
#         # final
#         pred = (pred_1+pre_2)/2
#         output = torch.argmax(pred)
#         if output == data.y[i]:
#             count += 1
#     return count / len(data)
