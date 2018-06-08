# coding: utf-8

import os
import jieba
import numpy as np 
from gensim.models import word2vec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

sentances = [
    "我有一个很久远的梦想它让我魂牵梦绕我希望有朝一日我可以中彩票", 
    "除此之外我还有一个特别小的梦想就是我希望这个世界和平"
]

word_count = []
for i in range(len(sentances)):
    word_count += [w for w in jieba.cut(sentances[i])]
word2index = {word:i for i, word in enumerate(set(word_count))}
# print(len(word2index), word2index)

## cbow model
def sen2bow(sentance, word2index):
    word_list = list(jieba.cut(sentance))
    bow = np.zeros(len(word2index))
    for w in word_list:
        bow[word2index[w]] = 1
    return bow
# print(sen2bow(sentances[0], word2index))

## word embeding example
idim, odim = len(word2index), len(word2index)//2
embed = nn.Embedding(idim,odim)
show_tensor = torch.LongTensor([word2index['和平'], word2index['梦想']])
test_embed = embed(Variable(show_tensor))
# print(test_embed)

## word2vec preprocess
raw_txt = list(jieba.cut(sentances[0] + sentances[1]))
data_hs = []    # data as (w, context)
for i in range(2, len(set(raw_txt))-2):
    target = [raw_txt[i-2], raw_txt[i-1], raw_txt[i+1], raw_txt[i+2]]
    contxt = raw_txt[i]
    data_hs.append((contxt, target))

data_ns = []    # data as (w, context[i])
raw_txt = list(jieba.cut(sentances[0] + sentances[1]))
for i in range(2, len(set(raw_txt))-2):
    target = [raw_txt[i-2], raw_txt[i-1], raw_txt[i+1], raw_txt[i+2]]
    contxt = raw_txt[i]
    for t in target:
        data.append((contxt, t))

# gensim package test
words = ''' 我沒有心
            我沒有真實的自我
            我只有消瘦的臉孔
            所謂軟弱
            所謂的順從一向是我
            的座右銘

            而我
            沒有那海洋的寬闊
            我只要熱情的撫摸
            所謂空洞
            所謂不安全感是我
            的墓誌銘

            而你
            是否和我一般怯懦
            是否和我一般矯作
            和我一般囉唆

            而你
            是否和我一般退縮
            是否和我一般肌迫
            一般地困惑

            我沒有力
            我沒有滿腔的熱火
            我只有滿肚的如果
            所謂勇氣
            所謂的認同感是我
            隨便說說

            而你
            是否和我一般怯懦
            是否和我一般矯作
            是否對你來說
            只是一場遊戲
            雖然沒有把握

            而你
            是否和我一般退縮
            是否和我一般肌迫
            是否對你來說
            只是逼不得已
            雖然沒有藉口'''

def seg_words(words, optf='segfile.txt'):
    seg_list = jieba.cut(words)
    with open(optf, 'w', encoding='UTF-8') as txt:
        for s in seg_list:
            txt.write(s+'\n')

def w2v_test(simw=u'是否', modsize=10, segf='segfile.txt'):
    sentences = word2vec.Text8Corpus(segf)
    mod = word2vec.Word2Vec(sentences, size=modsize)
    y = mod.most_similar(simw)
    print(y)
    os.remove(segf)

seg_words(words)
w2v_test()
# test end

class Word2Vec(nn.Module):
    def __init__(self, vob_size, embed_dim):
        self.embedding = nn.Embedding(vob_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vob_size)

    def forward(self, inputs):
        ebd = self.embedding(inputs).view((1, -1))
        opt = self.linear(ebd)
        plg = F.log_softmax(opt)    # probability by log softmax
        return plg

class RNN(nn.Module):
    def __init__(self, batch_size, token_count, embed_dim, hidden):
        self.batch_size = batch_size
        self.token_count = token_count
        self.embed_dim = embed_dim
        self.hidden = hidden

        self.lookup = nn.Embedding(token_count, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden)

    def forward(self, embed, state_word):
        embeded = self.lookup(embed)
        opt_word, state_word = self.rnn(embeded, state_word)
        return opt_word, state_word

    def init_hidden(self):
        return Variable(torch.zeros(1, self.batch_size, self.hidden))

class LSTM(RNN):
    """docstring for LSTM"""
    def __init__(self, batch_size, token_count, embed_dim, hidden):
        self.batch_size = batch_size
        self.token_count = token_count
        self.embed_dim = embed_dim
        self.hidden = hidden

        self.lookup = nn.Embedding(token_count, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden)

    def forward(self, embed, state_word):
        embeded = self.lookup(embed)
        opt_word, state_word = self.rnn(embeded, state_word)
        return opt_word, state_word

    def init_hidden(self):
        return Variable(torch.zeros(1, self.batch_size, self.hidden))
        
class BiLSTM(RNN):
    """docstring for BiLSTM"""
    def __init__(self, batch_size, token_count, embed_dim, hidden):
        self.batch_size = batch_size
        self.token_count = token_count
        self.embed_dim = embed_dim
        self.hidden = hidden

        self.lookup = nn.Embedding(token_count, embed_dim)
        self.rnn = nn.LSTM(embed_size, hidden, bidirectional= True)

    def forward(self, embed, state_word):
        embeded = self.lookup(embed)
        opt_word, state_word = self.rnn(embeded, state_word)
        return opt_word, state_word

    def init_hidden(self):
        return Variable(torch.zeros(1, self.batch_size, self.hidden))
        