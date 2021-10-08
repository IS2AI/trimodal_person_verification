#! /usr/bin/python
# -*- encoding: utf-8 -*-

from models.ResNetSE34L import *
from models.ResNetBlocks import *
import numpy

class ResNet(nn.Module):

    def __init__(self, block, layers, num_filters, nOut=512, **kwargs):
        super(ResNet, self).__init__()
        print('Embedding size is %d' % (nOut))
        self.inplanes = num_filters[0]
        self.num_images = kwargs["num_images"]

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filters[3], nOut)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, eval_mode):

        #x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)  # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        x = self.layer1(x)  # 56x56
        x = self.layer2(x)  # 28x28
        x = self.layer3(x)  # 14x14
        x = self.layer4(x)  # 7x7

        x = self.avgpool(x)  # 1x1
        x = torch.flatten(x, 1)  # remove 1 X 1 grid and make vector of tensor shape
        x = self.fc(x)

        # in the training mode one embedding per utterance by averaging all image feature vectors for the same utterance
        if not eval_mode:
            x_reshaped = x.reshape(int(x.shape[0] / self.num_images), self.num_images, -1)
            x = x_reshaped.mean(dim=1)

        return x


class SFPVNet(nn.Module):
    def __init__(self, nOut=256, **kwargs):
        super(SFPVNet, self).__init__()
        self.modality = kwargs["modality"].lower()
        self.nOut = nOut
        self.filters = kwargs["filters"]

        if "wav" in self.modality:
            self.aud_enc = ResNetSE(SEBasicBlock, [3, 4, 6, 3], self.filters, nOut, **kwargs)

        if "rgb" in self.modality:
            self.rgb_enc = ResNet(BasicBlock, [3, 4, 6, 3], self.filters, nOut, **kwargs)

        if "thr" in self.modality:
            self.thr_enc = ResNet(BasicBlock, [3, 4, 6, 3], self.filters, nOut, **kwargs)

        if "wav" in self.modality and "thr" in self.modality and "rgb" in self.modality:
            self.fc = nn.Linear(nOut * 3, 3)
            self.softmax = nn.Softmax(dim=1)

        elif "wav" in self.modality and "rgb" in self.modality:
            self.fc = nn.Linear(nOut * 2, 2)
            self.softmax = nn.Softmax(dim=1)

        elif "wav" in self.modality and "thr" in self.modality:
            self.fc = nn.Linear(nOut * 2, 2)
            self.softmax = nn.Softmax(dim=1)



    def forward(self, x, eval_mode):
        if "wav" in self.modality and "rgb" in self.modality and "thr" in self.modality:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]

            x1 = self.aud_enc(x1)
            x2 = self.rgb_enc(x2, eval_mode)
            x3 = self.thr_enc(x3, eval_mode)

            e = torch.cat((x1, x2, x3), 1)
            a = self.fc(e)
            alpha = self.softmax(a)
            x = torch.mul(e[:, :x1.shape[1]].T, alpha[:, 0]).T
            x = x + torch.mul(e[:, x1.shape[1]:x2.shape[1]+x1.shape[1]].T, alpha[:, 1]).T
            x = x + torch.mul(e[:, x2.shape[1]+x1.shape[1]:].T, alpha[:, 2]).T

        elif "wav" in self.modality and "rgb" in self.modality:
            x1 = x[0]
            x2 = x[1]
            x1 = self.aud_enc(x1)
            x2 = self.rgb_enc(x2, eval_mode)

            e = torch.cat((x1, x2), 1)
            a = self.fc(e)
            alpha = self.softmax(a)
            x = torch.mul(e[:, :x1.shape[1]].T, alpha[:, 0]).T
            x = x + torch.mul(e[:, x2.shape[1]:].T, alpha[:, 1]).T

        elif "wav" in self.modality and "thr" in self.modality:
            x1 = x[0]
            x2 = x[1]
            x1 = self.aud_enc(x1)
            x2 = self.thr_enc(x2, eval_mode)

            e = torch.cat((x1, x2), 1)
            a = self.fc(e)
            alpha = self.softmax(a)
            x = torch.mul(e[:, :x1.shape[1]].T, alpha[:, 0]).T
            x = x + torch.mul(e[:, x2.shape[1]:].T, alpha[:, 1]).T

        elif "wav" in self.modality:
            x = self.aud_enc(x)

        elif "rgb" in self.modality:
            x = self.rgb_enc(x, eval_mode)

        elif "thr" in self.modality:
            x = self.thr_enc(x, eval_mode)
        return x

def MainModel(nOut=256, **kwargs):
    model = SFPVNet(nOut, **kwargs)
    return model
