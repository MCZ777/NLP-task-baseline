import os
import torch
import keras as kr
from torch import nn
from cnews_loader import read_category,read_vocab,read_file,process_file
from textcnn_model import TextCNN
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


base_dir = 'cnews'
vocab_dir = os.path.join(base_dir,'cnews.vocab.txt')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CnnModel:
    def __init__(self):
        self.categoties,self.cat_to_id = read_category()
        self.words,self.word_to_id = read_vocab(vocab_dir)
        self.model = TextCNN()
        self.model.load_state_dict(torch.load('model_params.pkl'))

    def predict(self,content):
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        data = kr.preprocessing.sequence.pad_sequences([data],600)
        data = torch.LongTensor(data)
        y_pred = self.model(data)
        print(y_pred)
        class_index = torch.argmax(y_pred[0]).item()
        return self.categoties[class_index]




if __name__ == '__main__':
    test_dir = os.path.join(base_dir, 'cnews.test.txt')
    #获取文本的类别及其对应id的字典
    categories, cat_to_id = read_category()
    #获取训练文本中所有出现过的字及其所对应的id
    words, word_to_id = read_vocab(vocab_dir)
    #获取字数
    vocab_size = len(words)
    print(vocab_size)

    print('读取测试数据:')
    x_test,y_test = read_file(test_dir)
    # x_test_id,y_test_id = process_file(test_dir,word_to_id,cat_to_id,600)
    # print('测试标签:')
    # print(y_test[0:20])

    model = CnnModel()
    y_pre = []
    for i in tqdm(x_test,desc='预测测试集:'):
        y_hat = model.predict(i)
        # y_hat = y_hat.numpy()
        # y_pre.extend(np.argmax(y_hat,axis=1))#求每一行的最大值索引
        y_pre.append(y_hat)
    # print('预测标签:')
    # print(y_pre[0:20])
    print('F1-Score:{:.4f}'.format(f1_score(y_test,y_pre,average='micro')))

