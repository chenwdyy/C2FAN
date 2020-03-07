import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import network_v1 as network
from utils import loss
from utils import pre_process as prep
from torch.utils.data import DataLoader
from utils import lr_schedule
from utils import data_list
from utils.data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
import time

def image_classification_test(loader, model, test_10crop=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=4)

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()  # 加载基础网络结构
    parameter_list = base_network.get_parameters()
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    start_time = time.time()
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            # 在这里进行测试的工作
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                                                 base_network, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)
            end_time = time.time()
            print('iter {} cost time {:.4f} sec.'.format(i, end_time - start_time))  # 打印时间间隔
            start_time = time.time()

        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                                                             "iter_{:05d}_model.pth.tar".format(i)))

        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)  # 训练模式
        optimizer = lr_scheduler(optimizer, i, **schedule_param)#学习速率的改变

        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        inputs_source, labels_source = iter_source.next()
        inputs_source,  labels_source = inputs_source.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)  # 源域的分类损失
        classifier_loss.backward()
        optimizer.step()
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office',
                        choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='./data/office/amazon_list.txt',
                        help="The source dataset path list amazon_list, dslr_list,webcam_list")
    parser.add_argument('--t_dset_path', type=str, default='./data/office/webcam_list.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=100, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=2000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='./results_SourceOnly/office31/A2W_0915',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    # train config

    config = {}
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 10000
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = args.output_dir#这里就是输出的文件路径
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop": False, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": 1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": args.bottleneck_dim, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": args.bottleneck_dim,
                                        "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": args.bottleneck_dim,
                                        "new_cls": True}}

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": args.batch_size}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": args.batch_size}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": args.batch_size}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()

    seed_torch(0)  # 设置固定的随机种子，保持结果的一致性
    train(config)
