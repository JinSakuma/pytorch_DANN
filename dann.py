import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
from tqdm import tqdm

from model import Extractor, Classifier, Discrimimator


class DANN():
    def __init__(self, data_dict, log_dir='./log'):
        self.gamma = 10
        self.theta = 0.1

        self._build_models()
        self.cls_criterion = nn.NLLLoss()
        self.dom_criterion = nn.NLLLoss()
        self.optimizer = torch.optim.SGD([{'params': self.extractor.parameters()},
                                          {'params': self.classifier.parameters()},
                                          {'params': self.discriminator.parameters()}], lr=0.01, momentum=0.9)

        self.loader_trainS = data_dict['source:train']
        self.loader_testS = data_dict['source:test']
        self.loader_trainT = data_dict['target:train']
        self.loader_testT = data_dict['target:test']

        self.log_dir = log_dir

    def _build_models(self):
        gpu_flag = torch.cuda.is_available()

        extractor = Extractor()
        classifier = Classifier()
        discriminator = Discrimimator()

        if gpu_flag:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.extractor = extractor.to(self.device)
        self.classifier = classifier.to(self.device)
        self.discriminator = discriminator.to(self.device)

    def train_source_only(self, nEpochs=1, nSteps=1000, val_Steps=100):

        print("Train Source Only")
        for epoch in range(nEpochs):
            # train
            self.extractor.train()
            self.classifier.train()
            running_loss = 0
            cls_cnt = 0
            total = 0
            for batch_idx, dataS in zip(tqdm(range(nSteps)), self.loader_trainS):
                xS, yS = dataS
                xS, yS = Variable(xS.to(self.device)), Variable(yS.to(self.device))

                self.optimizer.zero_grad()
                featS = self.extractor(xS)
                cls_pred = self.classifier(featS)
                cls_loss = self.cls_criterion(cls_pred, yS)
                loss = cls_loss
                loss.backward()

                running_loss += loss.item()
                self.optimizer.step()

                cls_out = torch.argmax(cls_pred.data, dim=1)
                cls_cnt += (cls_out == yS.data).sum()
                total += yS.size(0)

            loss_train = running_loss / nSteps
            acc_cls = float(cls_cnt) / total

            # valid
            self.extractor.eval()
            self.classifier.eval()
            running_loss = 0
            cls_cntS = 0
            cls_cntT = 0
            val_Steps = 100
            total = 0
            for batch_idx, dataS, dataT in zip(tqdm(range(val_Steps)), self.loader_testS, self.loader_testT):
                xS, yS = dataS
                xT, yT = dataT

                xS, yS = Variable(xS.to(self.device)), Variable(yS.to(self.device))
                xT, yT = Variable(xT.to(self.device)), Variable(yT.to(self.device))

                self.optimizer.zero_grad()

                featS = self.extractor(xS)
                featT = self.extractor(xT)

                cls_predS = self.classifier(featS)
                cls_lossS = self.cls_criterion(cls_predS, yS)
                cls_predT = self.classifier(featT)
                cls_lossT = self.cls_criterion(cls_predT, yT)

                loss = (cls_lossS + cls_lossT) / 2

                running_loss += loss.item()
                cls_outS = torch.argmax(cls_predS.data, dim=1)
                cls_cntS += (cls_outS == yS.data).sum()
                cls_outT = torch.argmax(cls_predT.data, dim=1)
                cls_cntT += (cls_outT == yT.data).sum()
                total += yS.size(0)

            loss_val = running_loss / val_Steps
            acc_clsS = float(cls_cntS) / total
            acc_clsT = float(cls_cntT) / total

            print("<Train>")
            print('loss: {:.2f}'.format(loss_train))
            print('Source acc: {:.2f}%'.format(acc_cls*100.))

            print("<Val>")
            print('loss: {:.2f}'.format(loss_val))
            print('Source acc: {:.2f}%, Target acc: {:.2f}%'.format(acc_clsS*100., acc_clsT*100.))

        torch.save(self.extractor.state_dict(), os.path.join(self.log_dir, 'extractor_source_only_{:.1f}.pth').format(acc_clsT*100.))
        torch.save(self.classifier.state_dict(), os.path.join(self.log_dir, 'classifier_source_only_{:.1f}.pth').format(acc_clsT*100.))

    def train(self, nEpochs=20, nSteps=100, val_Steps=100):
        print('Train DANN')
        prev = 0.
        for epoch in range(nEpochs):
            # set params
            p = float(epoch) / nEpochs
            hp_lambda = 2. / (1. + np.exp(-self.gamma * p)) - 1
            lr = 0.01 / (1. + 10 * p) ** 0.75
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # train
            self.extractor.train()
            self.classifier.train()
            self.discriminator.train()
            running_loss = 0
            running_cls_loss = 0
            running_dom_loss = 0
            cls_cnt = 0
            total = 0
            for batch_idx, dataS, dataT in zip(tqdm(range(nSteps)), self.loader_trainS, self.loader_trainT):
                xS, yS = dataS
                xT, yT = dataT

                xS, yS = Variable(xS.to(self.device)), Variable(yS.to(self.device))
                xT, yT = Variable(xT.to(self.device)), Variable(yT.to(self.device))
                dS = Variable(torch.zeros(len(yS)).type(torch.LongTensor).to(self.device))
                dT = Variable(torch.ones(len(yT)).type(torch.LongTensor).to(self.device))

                self.optimizer.zero_grad()

                featS = self.extractor(xS)
                featT = self.extractor(xT)

                cls_pred = self.classifier(featS)
                cls_loss = self.cls_criterion(cls_pred, yS)

                dom_predS = self.discriminator(featS, hp_lambda)
                dom_predT = self.discriminator(featT, hp_lambda)
                dom_lossS = self.dom_criterion(dom_predS, dS)
                dom_lossT = self.dom_criterion(dom_predT, dT)
                dom_loss = dom_lossS + dom_lossT

                loss = cls_loss + self.theta * dom_loss
                loss.backward()

                running_loss += loss.item()
                running_cls_loss += cls_loss.item()
                running_dom_loss += dom_loss.item()
                self.optimizer.step()

                cls_out = torch.argmax(cls_pred.data, dim=1)
                cls_cnt += (cls_out == yS.data).sum()
                total += yS.size(0)

            loss_train = running_loss / nSteps
            cls_loss_train = running_cls_loss / nSteps
            dom_loss_train = running_dom_loss / nSteps

            acc_cls = float(cls_cnt) / total

            # valid
            self.extractor.eval()
            self.classifier.eval()
            self.discriminator.eval()
            running_loss = 0
            running_cls_loss = 0
            running_dom_loss = 0
            cls_cntS = 0
            cls_cntT = 0
            dom_cntS = 0
            dom_cntT = 0
            total = 0
            for batch_idx, dataS, dataT in zip(tqdm(range(val_Steps)), self.loader_testS, self.loader_testT):
                xS, yS = dataS
                xT, yT = dataT

                xS, yS = Variable(xS.to(self.device)), Variable(yS.to(self.device))
                xT, yT = Variable(xT.to(self.device)), Variable(yT.to(self.device))
                dS = Variable(torch.zeros(len(yS)).type(torch.LongTensor).to(self.device))
                dT = Variable(torch.ones(len(yT)).type(torch.LongTensor).to(self.device))

                self.optimizer.zero_grad()

                featS = self.extractor(xS)
                featT = self.extractor(xT)

                cls_predS = self.classifier(featS)
                cls_lossS = self.cls_criterion(cls_predS, yS)
                cls_predT = self.classifier(featT)
                cls_lossT = self.cls_criterion(cls_predT, yT)

                dom_predS = self.discriminator(featS, 1)
                dom_predT = self.discriminator(featT, 1)
                dom_lossS = self.dom_criterion(dom_predS, dS)
                dom_lossT = self.dom_criterion(dom_predT, dT)
                dom_loss = dom_lossS + dom_lossT

                loss = (cls_lossS + cls_lossT) / 2 + self.theta * dom_loss

                running_loss += loss.item()
                running_cls_loss += cls_loss.item()
                running_dom_loss += dom_loss.item()

                cls_outS = torch.argmax(cls_predS.data, dim=1)
                cls_cntS += (cls_outS == yS.data).sum()
                cls_outT = torch.argmax(cls_predT.data, dim=1)
                cls_cntT += (cls_outT == yT.data).sum()

                dom_outS = torch.argmax(dom_predS.data, dim=1)
                dom_cntS += (dom_outS == dS.data).sum()
                dom_outT = torch.argmax(dom_predT.data, dim=1)
                dom_cntT += (dom_outT == dT.data).sum()

                total += yS.size(0)

            loss_val = running_loss / val_Steps
            cls_loss_val = running_cls_loss / val_Steps
            dom_loss_val = running_dom_loss / val_Steps

            acc_clsS = float(cls_cntS) / total
            acc_clsT = float(cls_cntT) / total
            acc_dom = float(dom_cntS+dom_cntS) / (2 * total)

            print('[Epoch: {}]'.format(epoch))
            print('p: {:.2f}, lr: {:.2e}, l: {:.2f}'.format(p, lr, hp_lambda))

            print('<Train>')
            print('loss: {:.2f}, cls loss: {:.2f}, dom loss: {:.2f}'.format(loss_train, cls_loss_train, dom_loss_train))
            print('Source acc: {:.2f}%'.format(acc_cls*100.))

            print('<Val>')
            print('loss: {:.2f}, cls loss: {:.2f}, dom loss: {:.2f}'.format(loss_val, cls_loss_val, dom_loss_val))
            print('Source acc: {:.2f}%, Target acc: {:.2f}%, Domain acc: {:.2f}%'.format(acc_clsS*100., acc_clsT*100., acc_dom*100.))

            if acc_clsT > prev:
                prev = acc_clsT
                torch.save(self.extractor.state_dict(), os.path.join(self.log_dir, 'extractor_dann_{:.1f}.pth').format(acc_clsT*100.))
                torch.save(self.classifier.state_dict(), os.path.join(self.log_dir, 'classifier_dann_{:.1f}.pth').format(acc_clsT*100.))
