import torch.nn as nn
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

class PRO_network(nn.Module):
    def __init__(self, ninput, noutput=1):
        super(PRO_network, self).__init__()
        hidden_dim = 20
        self.layers = nn.Sequential(nn.Linear(ninput,hidden_dim),nn.ReLU(inplace=True))
        self.output = nn.Linear(hidden_dim*2,1)

    def forward(self, x):
        half_len = x.size(1)//2
        x1 = x[:,:half_len]
        x2 = x[:,half_len:]
        x1 = self.layers(x1)
        x2 = self.layers(x2)
        x = torch.cat([x1,x2],dim=1)
        x = self.output(x)
        return x.squeeze(-1)

class MAE_loss(nn.Module):
    def __init__(self):
        super(MAE_loss,self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred-y_true))


def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    """
    generate data pair
    """
    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim*2))
    training_labels = []
    for i in range(batch_size):
        if(i%4==0 or i%4==2):
            inner_index = rng.choice(inlier_indices, 1)[0]
            outer_index = rng.choice(outlier_indices,1)[0]
            inner_data = x_train[inner_index]
            outer_data = x_train[outer_index]
            # 混合数据
            if i%4==0:
                data = np.concatenate([inner_data,outer_data])
            else:
                data = np.concatenate([outer_data,inner_data])
            training_labels += [4]
        # 纯正常数据
        elif i%4==1:
            inner_index1 = rng.choice(inlier_indices, 1)[0]
            inner_index2 = rng.choice(inlier_indices, 1)[0]
            inner_data1 = x_train[inner_index1]
            inner_data2 = x_train[inner_index2]
            # 混合数据
            data = np.concatenate([inner_data1,inner_data2])
            training_labels += [0]

        elif i%4==3:
            outer_index1 = rng.choice(outlier_indices, 1)[0]
            outer_index2 = rng.choice(outlier_indices, 1)[0]
            outer_data1 = x_train[outer_index1]
            outer_data2 = x_train[outer_index2]
            # 混合数据
            data = np.concatenate([outer_data1, outer_data2])
            training_labels += [8]
        else:
            raise NameError("")
        ref[i] = data
    return np.array(ref), np.array(training_labels)

def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    return roc_auc, ap

# predict test data
def predict_test_data(model,test_X,train_X,trian_Y):
    train_inner_index_list  = np.where(train_Y==0)[0]
    train_outer_index_list = np.where(train_Y==1)[0]
    assert len(train_inner_index_list)+len(train_outer_index_list)==len(train_Y)
    E = 30
    result = []
    for i in range(len(test_X)):
        now_data = test_X[i]
        X = np.zeros((2*E,2*test_X.shape[1]))
        for i in range(2*E):
            if i%2==0:
                other_index = np.random.choice(train_outer_index_list,1)[0]
            else:
                other_index = np.random.choice(train_inner_index_list,1)[0]

            data = np.concatenate([now_data,train_X[other_index]])
            X[i] = data
        X = torch.tensor(X,dtype=torch.float32,device=DEVICE)
        Y = model(X)
        result.append(np.mean(Y.cpu().detach().numpy()))
    return np.array(result)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 666
MAX_INT = np.iinfo(np.int32).max
# we can assume how many anomaly data we know
KNOW_ANOMALY_NUM = 5
BATCH_SIZE = 512
if __name__ == "__main__":
    # X is input Y is label,you can load your data form csv or other formats
    X = np.random.rand(100,10)
    Y = np.zeros(100)
    # we assume there 100 are datas, and the data 0-9 are label 1(anomaly)
    Y[[0,1,2,3,4,5,6,7,8,9]] = 1
    train_X, test_X, train_Y, test_Y = \
        train_test_split(X, Y, test_size=0.2,
                        random_state=42, stratify = Y)
    outlier_indices = np.where(train_Y == 1)[0]
    inlier_indices = np.where(train_Y == 0)[0]
    random_generator = np.random.RandomState(RANDOM_SEED)
    # remove excess known outliers
    if len(outlier_indices)>KNOW_ANOMALY_NUM:
        remove_num = len(outlier_indices)-KNOW_ANOMALY_NUM
        remove_idx = random_generator.choice(outlier_indices,remove_num,replace=False)
        train_X = np.delete(train_X, remove_idx, axis=0)
        train_Y = np.delete(train_Y, remove_idx, axis=0)

    outlier_indices = np.where(train_Y == 1)[0]
    inlier_indices = np.where(train_Y == 0)[0]
    print("train data num:{} outlier num:{}".format(len(train_X),len(outlier_indices)))
    PRO_model = PRO_network(train_X.shape[1],1).to(DEVICE)
    criterion = MAE_loss()
    optimizer = torch.optim.RMSprop(PRO_model.parameters(),
                                    lr=1e-3,weight_decay=0.01)
    for e in range(50):
        PRO_model.train()
        train_loss = 0
        train_num = 0
        for nb in range(20):
            rng = np.random.RandomState(random_generator.randint(MAX_INT, size=1))
            # generate data pair
            X, Y = input_batch_generation_sup(train_X, outlier_indices,
                                              inlier_indices, BATCH_SIZE, rng)

            X = torch.tensor(X,dtype=torch.float32,device=DEVICE)
            Y = torch.tensor(Y,device=DEVICE)
            optimizer.zero_grad()
            outputs = PRO_model(X)
            loss = criterion(outputs, Y)
            if loss.item() == np.nan:
                raise NameError
            loss.backward()
            optimizer.step()
            train_loss += loss * len(outputs)
            train_num += len(outputs)

        PRO_model.eval()
        # 预估分数
        test_score = predict_test_data(PRO_model, test_X,train_X,train_Y)
        test_norm_score = test_score/8
        roc_auc,ap = aucPerformance(test_norm_score, test_Y)
        print("epoch:%d loss:%.4f AUC-ROC: %.4f, AUC-PR: %.4f" % (e,train_loss / train_num, roc_auc, ap))









