
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
Max_length = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextRNN(nn.Module):
    """文本分类，RNN模型"""

    def __init__(self):

        super(TextRNN, self).__init__()
        input_size = 3
        hidden_size = 512
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 三个待输入的数据、

        # self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size= self.hidden_size, num_layers=1, batch_first=True)
        # self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        # self.f1 = nn.Sequential(nn.Linear(256, 128),
        #                         nn.Dropout(0.8),
        #                         nn.ReLU())
        # self.f1 = nn.Sequential(nn.Linear(128, 2),
        #                         nn.Softmax())

        self.f0 = nn.Dropout(p=0.3)
        self.f1 = nn.Linear(self.hidden_size,2)
        self.f2 = nn.Sigmoid()

    def forward(self, x):
        # attn_weights = F.softmax(
        #     self.attn(torch.cat((x[0], hidden[0]), 1)), dim=1)
        # attn_applied = torch.bmm(attn_weights.unsqueeze(0),
        #                          encoder_outputs.unsqueeze(0))
        x = x.permute(0,2,1)
        output, (hn, cn)= self.rnn(x)
        output = hn

        # s, b, h = output.size()
        # output = output.view(s * b, h)
        # output = output[-1, :, :]
        output = self.f0(output)
        predict = self.f1(output)
        result = self.f2(predict)
        return result


class attCnn(nn.Module):

    def __init__(self, hidden_size, max_length = Max_length):
        self.hidden_size = hidden_size
        self.max_length = max_length

        super(attCnn, self).__int__()
        self.att = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.f0 = nn.Linear(self.hidden_size, 2)

    def forward(self, input, hidden):
        input = input.view(1,1,-1)
        attn_weights = F.softmax(
            self.att(torch.cat((input[0], hidden[0]), 1)), dim=1)
        atn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                input.unsqueeze(0))
        att_tem = torch.cat((input[0], atn_applied[0]),1)
        att_end = self.attn_combine(att_tem).unsqueeze(0)
        output = self.f0(att_end)
        return output


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(x.contiguous().view(-1, feature_dim),
                       self.weight).view(-1, step_dim)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class cnnAtt(nn.Module):
    def __init__(self, feature_num, hidden_length):

        super(cnnAtt, self).__init__()
        num_filters = 128
        self.att = Attention(feature_num, num_filters)
        self.f0 = nn.Dropout(p=0.1)
        self.f1 = nn.Linear(hidden_length, 2)
        self.f2 = nn.Sigmoid()

    def forward(self, input):
        att = self.att(input)
        att = self.f0(att)
        output = self.f1(att)
        output = self.f2(output)
        return output