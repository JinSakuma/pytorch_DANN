import torch
import torchvision
import torchvision.transforms as transforms
from dann import DANN
import os
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-l', dest='log_dir', default='./log')
    return parser.parse_args()


if __name__ == '__main__':
    data_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_dataS = torchvision.datasets.ImageFolder(root='./data/MNIST/train',
                                                   transform=data_transform)
    loader_trainS = torch.utils.data.DataLoader(train_dataS,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=4)

    test_dataS = torchvision.datasets.ImageFolder(root='./data/MNIST/test',
                                                  transform=data_transform)
    loader_testS = torch.utils.data.DataLoader(train_dataS,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=4)

    train_dataT = torchvision.datasets.ImageFolder(root='./data/MNIST-M/train',
                                                   transform=data_transform)
    loader_trainT = torch.utils.data.DataLoader(train_dataT,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=4)

    test_dataT = torchvision.datasets.ImageFolder(root='./data/MNIST-M/test',
                                                  transform=data_transform)
    loader_testT = torch.utils.data.DataLoader(train_dataT,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=4)

    data_dict = {'source:train': loader_trainS,
                 'source:test': loader_testS,
                 'target:train': loader_trainT,
                 'target:test': loader_testT}

    args = arg_parser()
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    DANN = DANN(data_dict, log_dir)
    DANN.train_source_only(nEpochs=1, nSteps=1000, val_Steps=100)
    DANN.train(nEpochs=20, nSteps=100, val_Steps=100)
