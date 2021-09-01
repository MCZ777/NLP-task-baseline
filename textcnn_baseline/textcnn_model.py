import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class TextCNN(nn.Module):
    """CNN模型"""

    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(5000, 64)
        self.conv = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=256, kernel_size=5),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=596))
        self.f1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.embedding(x)  # batch_size,length,embedding_size  64*600*64
        x = x.permute(0, 2, 1)  # 将tensor的维度换位。64*64*600

        # batch_size,卷积核个数out_channels，(句子长度-kernel_size)/步长+1
        x = self.conv(x)  # Conv1后64*256*596,ReLU后不变,MaxPool1d后64*256*1;

        x = x.view(-1, x.size(1))  # 64*256
        x = F.dropout(x, 0.8)
        x = self.f1(x)  # 64*10 batch_size * class_num
        return x

if __name__ == '__main__':
    net = TextCNN()
    print(net)
