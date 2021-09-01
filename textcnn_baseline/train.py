import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import os
from tqdm import tqdm
from textcnn_model import TextCNN
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(model, Loss, x_val, y_val):
    """测试集准确率评估"""
    batch_val = batch_iter(x_val, y_val, 64)
    acc = 0
    los = 0
    for x_batch, y_batch in batch_val:
        size = len(x_batch)
        x = np.array(x_batch)  # 创建数组
        y = np.array(y_batch)
        x = torch.LongTensor(x) # 构建long类型的张量
        x = x.to(device)
        y = torch.Tensor(y)
        y = y.to(device)

        # model.to(device)
        out = model(x)
        loss = Loss(out, y)

        loss_value = np.mean(loss.detach().cpu().numpy())  # detach()复制一个tensor，可以自动求导
        accracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).cpu().numpy())
        acc += accracy * size
        los += loss_value * size
    return los / len(x_val), acc / len(x_val)


base_dir = 'cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')


def train():
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, 600)  # 获取训练数据每个字的id和对应标签
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, 600)

    model = TextCNN()
    model.to(device)

    # 选择损失函数
    Loss = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 0
    for epoch in tqdm(range(100),desc='Train:'):
        i = 0
        print('epoch:{}'.format(epoch))
        batch_train = batch_iter(x_train, y_train, 64)
        for x_batch, y_batch in batch_train:
            i += 1
            x = np.array(x_batch)
            y = np.array(y_batch)
            x = torch.LongTensor(x)  # 构建long类型的张量
            x = x.to(device)
            y = torch.Tensor(y)
            y = y.to(device)

            optimizer.zero_grad()  # 梯度初始化为零
            out = model(x)  # 前向传播求出预测的值
            loss = Loss(out, y)  # 求loss
            loss.backward()  # 反向传播求梯度
            optimizer.step()  # 更新所有参数

            #对模型进行验证
            if i % 90 == 0:
                los , accracy = evaluate(model,Loss,x_val,y_val)
                print('loss:{},accracy:{}'.format(los,accracy))
                if accracy > best_val_acc:
                    torch.save(model.state_dict(),'model_params.pkl')
                    best_val_acc = accracy



if __name__ == '__main__':
    # 获取文本的类别及其对应id的字典
    categories, cat_to_id = read_category()
    # 获取训练文本中所有出现过的字及其所对应的id
    words, word_to_id = read_vocab(vocab_dir)
    # 获取字数
    vocab_size = len(words)
    print('train')
    train()
