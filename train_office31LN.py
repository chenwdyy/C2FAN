import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models.network_1022 as network
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
import time
from torch.optim import lr_scheduler
import torch.nn.functional as F
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


#测试分类正确率
def image_classification_test_1(loader, model, test_10crop=False):
    start_test = True
    with torch.no_grad():
        if test_10crop:#测试集增强技术
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
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

def get_entropy_loss(p_softmax):  # 这个是熵的损失
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))

def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
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


    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()  #加载基础网络结构
    parameter_list = base_network.get_parameters()
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))

    crit=LabelSmoothingLoss(smoothing=0.1,classes=31)#标签平滑操作

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
    best_acc = 0.0
    start_time = time.time()
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            # 在这里进行测试的工作
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, base_network, test_10crop=False)
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
        ## train one iter
        base_network.train(True)  # 训练模式
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        #目标域的熵正则化操作
        t_logit=outputs_target
        t_prob = F.softmax(t_logit)
        t_entropy_loss = get_entropy_loss(t_prob)  # 计算目标域的熵的损失
        entropy_loss = config['ENT_w']*(t_entropy_loss)

        # classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)  # 源域的分类损失
        classifier_loss=crit(outputs_source,labels_source)

        total_loss = classifier_loss + entropy_loss
        if i % config["test_interval"] == config["test_interval"] - 1:
            print('total loss: {:.4f}, classifier loss: {:.4f}'.format(
                total_loss.item(),classifier_loss.item()
            ))

        total_loss.backward()
        optimizer.step()
        if (i+1) % 1000==0:
            model_path=os.path.join(os.path.dirname(config['save_model_name']),'temp_model.pth')
            save_model(base_network,model_path)

    model_path=config['save_model_name']
    best_model_path=os.path.join(os.path.dirname(model_path),'best_model.pth')
    save_model(base_network,model_path)
    save_model(best_model,best_model_path)

    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

def save_model(model,model_path):#保存模型参数
    torch.save(model.state_dict(),model_path)

def load_model(model,model_path):#加载保存的模型参数
    return model.load_state_dict(torch.load(model_path))

import numpy
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18","ResNet50", "ResNet101", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office',
                        choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='./data/office/amazon_list.txt',
                        help="The source dataset path list amazon_list, dslr_list,webcam_list")
    parser.add_argument('--t_dset_path', type=str, default='./data/office/webcam_list.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=100, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=20000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='./results/office31LN_1022/a2w_1022',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--save_model_name', type=str, default='./checkpoints/office31_1022/a2w/model_1022_9999.pth')
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--start_iter', type=int, default=299)
    parser.add_argument('--ENT_w', type=float, default=0.05)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    # train config
    config = {}
    config['lr'] = args.lr
    config['start_iter'] = args.start_iter
    config['ENT_w'] = args.ENT_w
    config['save_model_name']=args.save_model_name
    if not os.path.exists(os.path.dirname(args.save_model_name)):#创建保存模型的文件夹
        os.makedirs(os.path.dirname(args.save_model_name))
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 10000
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = args.output_dir
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
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}


    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    # config["optimizer"] = {"type": optim.Adam, "optim_params": {'lr': 0.0003,"weight_decay": 0.0005},
    #                        "lr_type": "inv", \
    #                        "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}


    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 32}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": 32}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": 32}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = args.lr # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = args.lr  # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = args.lr  # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = args.lr  # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = args.lr  # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()

    seed_torch(1000)  # 设置固定的随机种子，保持结果的一致性
    train(config)
