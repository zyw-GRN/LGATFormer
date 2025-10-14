import pickle
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
from utils import scRNADataset, load_data1, adj2saprse_tensor, Evaluation,  Network_Statistic
import pandas as pd
import numpy as np
import random
import glob
import os, sys
from scipy.sparse import dok_matrix, csc_matrix
import time
import argparse
sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from util_functions import *


parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, default=500, help='network scale')
parser.add_argument('--data', type=str, default='mDC', help='data type')
parser.add_argument('--net', type=str, default='Specific', help='network type')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--Type',type=str,default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--max-train-num', type=int, default=10000,
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--hop', default=2, metavar='S',
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=100,
                    help='if > 0, upper bound the # nodes per hop by subsampling')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)





def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)

lujing = 'Dataset/Benchmark Dataset/'+args.net+' Dataset/'+args.data+'/TFs+'+str(args.num)
tl = 'Dataset/Benchmark Dataset/Data/'+args.net+'/'+args.data+' '+str(args.num)
print(tl)
print(lujing)
exp_file = lujing+'/BL--ExpressionData.csv'
tf_file = lujing+'/TF.csv'
target_file = lujing+'/Target.csv'

train_file = tl+'/Train_set.csv'
val_file = tl+'/val_set_file'
test_file = tl+'/Test_set.csv'


tf_embed_path = r'Result/.../Channel1.csv'
target_embed_path = r'Result/.../Channel2.csv'


data_input = pd.read_csv(exp_file,index_col=0)
loader = load_data1(data_input)
feature = loader.exp_data()
tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
tf = tf.to(device)


train_data = pd.read_csv(train_file, index_col=0).values
validation_data = pd.read_csv(val_file, index_col=0).values
test_data = pd.read_csv(test_file, index_col=0).values


train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf, loop=args.loop)

A = adj.tocsc()

pos_train1 = train_data[train_data[:, 2] == 1]
neg_train1 = train_data[train_data[:, 2] == 0]
pos_val1 = validation_data[validation_data[:, 2] == 1]
neg_val1 = validation_data[validation_data[:, 2] == 0]
pos_test1 = test_data[test_data[:, 2] == 1]
neg_test1 = test_data[test_data[:, 2] == 0]

pos_train2 = np.vstack((pos_train1, pos_val1))
neg_train2 = np.vstack((neg_train1, neg_val1))

pos_train = (pos_train2[:, 0], pos_train2[:, 1])
neg_train = (neg_train2[:, 0], neg_train2[:, 1])
pos_test = (pos_test1[:, 0], pos_test1[:, 1])
neg_test = (neg_test1[:, 0], neg_test1[:, 1])

args.max_nodes_per_hop = 100
train_graphs, test_graphs, max_n_label = links2subgraphs(A, pos_train, neg_train, pos_test, neg_test, args.hop, args.max_nodes_per_hop, feature)
print(('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs))))
file_path = tl + "/train_lines_data.pkl"
if os.path.exists(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"DELETE:{file_path}")
    else:
        print(f"UNDELETE{file_path} ")
else:
    print(f"{file_path}not exit")

file_path = tl + "/test_lines_data.pkl"
if os.path.exists(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"DELETE:{file_path}")
    else:
        print(f"UNDELETE{file_path} ")
else:
    print(f"{file_path}not exit")

random.shuffle(train_graphs)
random.shuffle(test_graphs)
train_lines, ind_tr = to_linegraphs(train_graphs, max_n_label, tl, "/train_lines_data")
test_lines, ind_te = to_linegraphs(test_graphs, max_n_label, tl, "/test_lines_data")

dim_feature = feature.shape[1]
a = [max_n_label, dim_feature, ind_tr, ind_te]
with open(tl + "/num.pkl", "wb") as f:
    pickle.dump(a, f)

