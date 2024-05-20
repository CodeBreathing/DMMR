from preprocess import getDataLoaders
import math
from torch.utils.tensorboard import SummaryWriter
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

def main(data_loader_dict, args, optim_config, cuda, writer, one_subject, seed=3):
    set_seed(seed)
    if args.dataset_name == 'seed3':
        iteration = 7
    elif args.dataset_name == 'seed4':
        iteration = 3
    acc = trainDMMR_WithoutNoise(data_loader_dict, optim_config, cuda, args, iteration, writer, one_subject)
    return acc

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description='DMMR')

    #config of experiment
    parser.add_argument("--way", type=str, default='DMMR/seed3', help="name of current way")
    parser.add_argument("--index", type=str, default='0', help="tensorboard index")

    #config of dataset
    parser.add_argument("--dataset_name", type=str, nargs='?', default='seed3', help="the dataset name, supporting seed3 and seed4")
    parser.add_argument("--session", type=str, nargs='?', default='1', help="selected session")
    parser.add_argument("--subjects", type=int, choices=[15], default=15, help="the number of all subject")
    parser.add_argument("--dim", type=int, default=310, help="dim of input")

    #config of DMMR
    parser.add_argument("--input_dim", type=int, default=310, help="input dim is the same with sample's last dim")
    parser.add_argument("--hid_dim", type=int, default=64, help="hid dim is for hidden layer of lstm")
    parser.add_argument("--n_layers", type=int, default=1, help="num of layers of lstm")
    parser.add_argument("--epoch_fineTuning", type=int, default=500, help="epoch of the fine-tuning phase")
    parser.add_argument("--lr", type=int, default=1e-3, help="epoch of calModel")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay")
    parser.add_argument("--beta", type=float, default=0.05, help="balancing hyperparameter in the loss of pretraining phase")


    args = parser.parse_args()
    args.source_subjects = args.subjects-1
    args.seed3_path = "../eeg_data/ExtractedFeatures/"
    args.seed4_path = "../eeg_data/eeg_feature_smooth/"
    if cuda:
        args.num_workers_train = 4
        args.num_workers_test = 2
    else:
        args.num_workers_train = 0
        args.num_workers_test = 0
    if args.dataset_name == "seed3":
        args.path = args.seed3_path
        args.cls_classes = 3
        args.time_steps = 30
        args.batch_size = 512  #batch_size
        args.epoch_preTraining = 300  #epoch of the pre-training phase
    elif args.dataset_name == "seed4":
        args.path = args.seed4_path
        args.cls_classes = 4
        args.time_steps = 10
        args.batch_size = 256  #batch_size
        args.epoch_preTraining = 400  #epoch of the pre-training phase
    else:
        print("need to define the input dataset")
    optim_config = {"lr": args.lr, "weight_decay": args.weight_decay}
    # leave-one-subject-out cross-validation
    acc_list=[]
    writer = SummaryWriter("data/session"+args.session+"/"+args.way+"/" + args.index)
    for one_subject in range(0, args.subjects):
        # 1.data preparation
        source_loaders, test_loader = getDataLoaders(one_subject, args)
        data_loader_dict = {"source_loader": source_loaders, "test_loader":test_loader}
        # 2. main
        acc = main(data_loader_dict, args, optim_config, cuda, writer, one_subject)
        writer.add_scalars('single experiment acc: ',
                           {'test acc': acc}, one_subject + 1)
        writer.flush()
        acc_list.append(acc)
    writer.add_text('final acc avg', str(np.average(acc_list)))
    writer.add_text('final acc std', str(np.std(acc_list)))
    acc_list_str = [str(x) for x in acc_list]
    writer.add_text('final each acc', ",".join(acc_list_str))
    writer.add_scalars('final experiment acc scala: /avg',
                       {'test acc': np.average(acc_list)})
    writer.add_scalars('final experiment acc scala: /std',
                       {'test acc': np.std(acc_list)})
    writer.close()
