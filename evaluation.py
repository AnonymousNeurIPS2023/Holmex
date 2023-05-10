import pickle
from util import *

# options -------------------------------------------------------------------------------------------------------------
# set random seed
seed = 1234
setup_seed(seed)

task_str = "1"

cbm_store_path = 'CLIP_ViT_L_14'
mlp_store_path = 'CLIP_ViT_L_14'

# cbm_store_path = 'CLIP_ViT_B_32'
# mlp_store_path = 'CLIP_ViT_B_32'
#
# cbm_store_path = 'CLIP_ViT_B_32'
# mlp_store_path = 'CLIP_ViT_L_14'
#
# cbm_store_path = 'CLIP_ViT_L_14'
# mlp_store_path = 'CLIP_ViT_B_32'
#
# cbm_store_path = 'CLIP_ViT_L_14'
# mlp_store_path = 'resnet50'
#
# cbm_store_path = 'CLIP_ViT_B_32'
# mlp_store_path = 'resnet50'

# 'CLIP_ViT_L_14', 'CLIP_ViT_B_32', 'resnet50'
# ----------------------------------------------------------------------------------------------------------------------
f = open("tasks/task"+task_str+"/"+cbm_store_path+"/data_test.pickle", 'rb')
X, y = pickle.load(f)
test_data_cbm = MyData(X=torch.tensor(X, dtype=torch.float), y=torch.tensor(y, dtype=torch.int64))

f = open("tasks/task"+task_str+"/"+mlp_store_path+"/data_test.pickle", 'rb')
X, y = pickle.load(f)
test_data_mlp = MyData(X=torch.tensor(X, dtype=torch.float), y=torch.tensor(y, dtype=torch.int64))


# load models
cbm = pickle.load(open("tasks/task"+task_str+"/"+cbm_store_path+"/cbm.pickle", 'rb'))
cbm_clean = pickle.load(open("tasks/task"+task_str+"/"+cbm_store_path+"/cbm_clean.pickle", 'rb'))
mlp = pickle.load(open("tasks/task"+task_str+"/"+mlp_store_path+"/mlp.pickle", 'rb'))

print("cbm")
print(evaluation(cbm, test_data_cbm))
print("cbm_clean")
print(evaluation(cbm_clean, test_data_cbm))
print("MLP")
print(evaluation(mlp, test_data_mlp))

print("transfer edit with (0.5, 1, -0.5)")
print(transfer_evaluation(black_model=mlp, white_model=cbm, white_model_edit=cbm_clean, data_black=test_data_mlp,
                          data_white=test_data_cbm, weight=(0.5, 1, -0.5)))

print("transfer edit with (1, 1, -1)")
print(transfer_evaluation(black_model=mlp, white_model=cbm, white_model_edit=cbm_clean, data_black=test_data_mlp,
                          data_white=test_data_cbm, weight=(1, 1, -1)))

print("ensemble mlp + cbm")
print(ensemble_evaluation(model1=mlp, model2=cbm, data1=test_data_mlp, data2=test_data_cbm))

print("ensemble mlp + cbm_clean")
print(ensemble_evaluation(model1=mlp, model2=cbm_clean, data1=test_data_mlp, data2=test_data_cbm))

print("ensemble mlp + cbm_clean + cbm")
print(transfer_evaluation(black_model=mlp, white_model=cbm, white_model_edit=cbm_clean, data_black=test_data_mlp,
                          data_white=test_data_cbm, weight=(1/3, 1/3, 1/3)))

output_str = str(evaluation(cbm, test_data_cbm)) + "\t" + str(evaluation(cbm_clean, test_data_cbm))+ "\t" \
             + str(evaluation(mlp, test_data_mlp)) + "\t" + \
        str(transfer_evaluation(black_model=mlp, white_model=cbm, white_model_edit=cbm_clean, data_black=test_data_mlp,
                          data_white=test_data_cbm, weight=(0.5, 1, -0.5))) + "\t" \
+ str(transfer_evaluation(black_model=mlp, white_model=cbm, white_model_edit=cbm_clean, data_black=test_data_mlp,
                          data_white=test_data_cbm, weight=(1, 1, -1))) + "\t" \
+ str(ensemble_evaluation(model1=mlp, model2=cbm, data1=test_data_mlp, data2=test_data_cbm)) + "\t" \
+ str(ensemble_evaluation(model1=mlp, model2=cbm_clean, data1=test_data_mlp, data2=test_data_cbm)) + "\t" \
+ str(transfer_evaluation(black_model=mlp, white_model=cbm, white_model_edit=cbm_clean, data_black=test_data_mlp,
                          data_white=test_data_cbm, weight=(1/3, 1/3, 1/3)))

print(output_str)
