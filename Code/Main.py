import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import os, sys
sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_DGCNN.main import *
from util_functions import *
from model import Net
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--max-train-num', type=int, default=10000, help='set maximum number of train links (to fit into memory)')
parser.add_argument('--seed', type=int, default=128, metavar='S', help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.5, help='ratio of test links')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# model settings
parser.add_argument('--hop', default=2, metavar='S', help='enclosing subgraph hop number, options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=100, help='if > 0, upper bound the # nodes per hop by subsampling')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

gene = 'mHSC-GM'
tl = 'Dataset/Benchmark Dataset/Data/Specific/'+gene+' 500'
print(tl)
train_lines = []
test_lines = []
with open(tl+"/num.pkl", "rb") as f:
    a = pickle.load(f)
    max_n_label = a[0]
    dim_feature = a[1]
    hub_train = a[2]
    hub_test = a[3]

args.hidden = [128, 128, 64]
args.out_dim = 0
args.dropout = True
args.num_class = 2
args.mode = 'gpu'
args.num_epochs = 15
args.learning_rate = 0.0005
args.batch_size = 20
args.printAUC = True
args.feat_dim = (max_n_label + 1)*2 + dim_feature*2
args.attr_dim = 0
args.latent_dim = [32, 32, 32]
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classifier = Net(args.feat_dim, args.hidden, args.latent_dim, args.dropout)
if args.mode == 'gpu':
    classifier = classifier.to(device)
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

for epoch in range(args.num_epochs):
    classifier.train()
    avg_loss, AUC, AUPR, AUPR_norm = loop_dataset_gem(classifier, "train", tl, hub_train, args, optimizer=optimizer)
    print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f auroc %.5f aupr %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], AUC, AUPR)))

    classifier.eval()
    test_loss, AUC, AUPR, AUPR_norm = loop_dataset_gem(classifier, "test", tl, hub_test, args, None)
    print(('average test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f auroc %.5f aupr %.5f' % (epoch, test_loss[0], test_loss[1], test_loss[2], avg_loss[3], AUC, AUPR)))



























