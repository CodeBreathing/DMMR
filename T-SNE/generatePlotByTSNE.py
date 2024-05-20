from preprocess import getDataLoaders
import math
import argparse
from train import *
import random
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(data_loader_dict,args, cuda, seed=3):
    set_seed(seed)
    acc = TSNEForDMMR(data_loader_dict, cuda, args)
    return acc

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description='DMMR')

    #config of experiment
    parser.add_argument("--way", type=str, default='TSNE', help="name of current way")
    parser.add_argument("--index", type=str, default='plot features', help="tensorboard index")
    parser.add_argument("--num_workers_train", type=int, default=0, help="classes of dataset")
    parser.add_argument("--num_workers_test", type=int, default=0, help="classes of dataset")
    parser.add_argument("--path", type=str, default="../eeg_data/ExtractedFeatures/", help="classes of dataset")

    #config of dataset
    parser.add_argument("--dataset_name", type=str, nargs='?', default='seed3', help="all subject numbers")
    parser.add_argument("--session", type=str, nargs='?', default='1', help="select session")
    parser.add_argument("--subjects", type=int, choices=[15], default=15, help="all subject numbers")
    parser.add_argument("--cls_classes", type=int, choices=[3], default=3, help="classes of dataset")
    parser.add_argument("--dim", type=int, default=310, help="dim of input")

    #config of net
    parser.add_argument("--input_dim", type=int, default=310, help="input dim is the same with sample's last dim")
    parser.add_argument("--hid_dim", type=int, default=64, help="hid dim is for hidden layer of lstm")
    parser.add_argument("--n_layers", type=int, default=2, help="num of layers of lstm")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--time_steps", type=int, choices=[30], default=30, help="window size")
    parser.add_argument("--epoch_preTraining", type=int, default=300, help="epoch of baseModel")
    parser.add_argument("--epoch_fineTuning", type=int, default=500, help="epoch of baseModel4")
    parser.add_argument("--lr", type=int, default=1e-3, help="epoch of pretrain")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay")
    parser.add_argument("--beta", type=float, default=0.05, help="balancing hyperparameter in the loss of pretraining phase")

    args = parser.parse_args()
    args.source_subjects = args.subjects-1
    net_config = {"fts": args.dim, "cls": args.cls_classes}
    optim_config = {"lr": args.lr, "weight_decay": args.weight_decay}

    acc_list=[]
    #use the subject 1 as the target subject for testing
    one_subject = 1
    # 1.data preparation
    source_loaders, test_loader = getDataLoaders(one_subject, args)
    data_loader_dict = {"source_loader": source_loaders, "test_loader":test_loader}
    # 2. call main
    acc = main(data_loader_dict, args, cuda)