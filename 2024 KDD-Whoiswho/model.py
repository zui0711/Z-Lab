import torch.nn.functional as F
from torch import nn, optim
import torch
import torchaudio.transforms as T
import numpy as np

class FullFC(torch.nn.Module):
    def __init__(self, emb_dim, other_f_dim):
        super(FullFC, self).__init__()
        self.fc_emb1 = nn.Linear(emb_dim, 256)
        self.fc_emb2 = nn.Linear(256, 64)
        self.fc_emb3 = nn.Linear(64, 32)

        self.fc_other1 = nn.Linear(other_f_dim, 64)
        self.fc_other2 = nn.Linear(64, 32)

        self.fc = nn.Linear(64, 1)
        self.dropout0 = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        # self.fc1 = nn.Linear(32, 1)

    def forward(self, inputs, mode='train'):
        emb_x = inputs['emb']
        other_x = inputs['other']

        emb_x = F.relu(self.fc_emb1(emb_x))
        # emb_x = self.dropout0(emb_x)
        emb_x = F.relu(self.fc_emb2(emb_x))
        # emb_x = self.dropout0(emb_x)
        emb_x = F.relu(self.fc_emb3(emb_x))
        # emb_x = self.dropout(emb_x)
        # emb_x = emb_x * F.sigmoid(emb_x)

        other_x = F.relu(self.fc_other1(other_x))
        # other_x = self.dropout0(other_x)
        other_x = F.relu(self.fc_other2(other_x))
        # other_x = self.dropout(other_x)
        # other_x = other_x * F.sigmoid(other_x)

        x = self.fc(torch.concat([emb_x, other_x], dim=1))
        # prediction = self.dropout(prediction)
        # prediction = prediction * F.sigmoid(prediction)

        # prediction = self.fc1(F.relu(prediction))
        p1 = (self.dropout1(x))
        p2 = (self.dropout2(x))
        p3 = (self.dropout3(x))
        # prediction = (self.dropout1(x)+self.dropout2(x)+self.dropout3(x)) / 3
        # prediction = F.sigmoid(prediction)

        if mode == 'test':
            return (F.sigmoid(p1)+F.sigmoid(p2)+F.sigmoid(p3))/3
        else:
            # return self.cal_loss(prediction, inputs['label'])
            return self.cal_loss1(p1, p2, p3, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        loss = F.binary_cross_entropy(prediction.flatten(), label)
        # loss = F.nll_loss(prediction, label.long(), label_smoothing=0.05)
        return loss, prediction, label



    @staticmethod
    def cal_loss1(p1, p2, p3, label):
        loss1 = F.binary_cross_entropy_with_logits(p1.flatten(), label)
        loss2 = F.binary_cross_entropy_with_logits(p2.flatten(), label)
        loss3 = F.binary_cross_entropy_with_logits(p3.flatten(), label)
        # loss = F.nll_loss(prediction, label.long(), label_smoothing=0.05)
        return ((loss1+loss2+loss3)/3,
                (F.sigmoid(p1)+F.sigmoid(p2)+F.sigmoid(p3))/3, label)
