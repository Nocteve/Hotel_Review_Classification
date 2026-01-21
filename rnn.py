import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from pathlib import Path
import pandas as pd
import json 
import jieba
import logging
import torch.nn.functional as F
import numpy as np
from vose_sampler import VoseAlias
import random
import time
import matplotlib.pyplot as plt

train_loss=[]

start=time.time()

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

vocab_path=Path('./data.json')
vocab={}
with open(vocab_path,encoding='utf-8') as f:
    vocab=json.load(f)

class skip_gram_model(nn.Module):
    def __init__(self,dim=512,vocab_size=29715):
        super(skip_gram_model,self).__init__()
        self.in_embed=nn.Embedding(vocab_size,dim)
        self.out_embed=nn.Embedding(vocab_size,dim)
        self.sigmoid=nn.Sigmoid()
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
    def forward(self,x):
        center_word,context_word,n_sample_list=x
        center_vec=self.in_embed(center_word)
        context_vec=self.out_embed(context_word)
        loss=torch.tensor(0,dtype=torch.float,device=device)
        loss-=F.logsigmoid(torch.sum(center_vec*context_vec))

        n_sample_vecs=self.out_embed(n_sample_list)
        n_score_vec=torch.sum(n_sample_vecs*center_vec.unsqueeze(1),dim=2)
        loss-=torch.sum(F.logsigmoid(-n_score_vec))
        return loss

w2v_model=skip_gram_model()
w2v_model_path=Path('./final.pth')
w2v_model.load_state_dict(torch.load(w2v_model_path))
w2v_model.to(device)
w2v_model.eval()

class rnn_cell(nn.Module):
    def __init__(self):
        super(rnn_cell,self).__init__()
        self.w_fc=nn.Linear(512,512)
        self.u_fc=nn.Linear(512,512)
        self.tanh=nn.Tanh()
        # self.w_fc.weight.data.uniform_(-0.1, 0.1)
        # self.u_fc.weight.data.uniform_(-0.1, 0.1)

        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.w_fc.weight)
        nn.init.xavier_uniform_(self.u_fc.weight)
        nn.init.zeros_(self.w_fc.bias)
        nn.init.zeros_(self.u_fc.bias)

    def forward(self,x,h):
        output=self.w_fc(x)+self.u_fc(h)
        output=self.tanh(output)
        return output
    def zero_h(self):
        self.h=torch.zeros(512,dtype=torch.float,device=device)
class rnn(nn.Module):
    def __init__(self):
        super(rnn,self).__init__()
        self.layer_num=2
        self.rnn_cells=nn.ModuleList()
        self.load_rnn_cell()
        self.out_fc1=nn.Linear(512,128)
        self.out_fc2=nn.Linear(128,2)
        
        self.relu=nn.ReLU()   
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()     
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.2)
        self.embed_dropout=nn.Dropout(0.3)
        self.layer_norm=nn.LayerNorm(512)
    def load_rnn_cell(self):
        for _ in range(self.layer_num):
            cell=rnn_cell()
            self.rnn_cells.append(cell)
    def forward(self,input):
        h=[]
        h_sum=torch.zeros(512,device=device)
        for _ in range(self.layer_num):
            h.append(torch.zeros(512,device=device))   
        
        words=jieba.cut(input)
        #h=torch.zeros(512,dtype=torch.float,device=device)
        for word in words:
            word_num=vocab[word]
            word_vec=w2v_model.in_embed(torch.tensor(word_num,device=device))
            for cell_id,rnn_cell in enumerate(self.rnn_cells):
                if cell_id==0:
                    h[cell_id]=rnn_cell(word_vec,h[cell_id])
                else:
                    temp_h=self.dropout1(h[cell_id-1])#防止过拟合,层间dropout
                    h[cell_id]=rnn_cell(temp_h,h[cell_id])
                    
                    # h[cell_id]=rnn_cell(h[cell_id-1],h[cell_id])

                    h[cell_id]=self.layer_norm(h[cell_id])#归一化(非第一层进行)

        y=self.out_fc1(h[-1])
        y=self.sigmoid(y)
        y=self.dropout2(y)
        y=self.out_fc2(y) 
        return y

class htl_data(Dataset):
    def __init__(self):
        super(htl_data,self).__init__()
        self.data_path=Path('./ChnSentiCorp_htl_all.csv')
        self.dataset=self.load_data()
    def load_data(self):
        data_frame=pd.read_csv(self.data_path)
        data_texts=data_frame['review']
        data_labels=data_frame['label']
        dataset=[]
        for i in range(len(data_texts)):
            if isinstance(data_texts[i],str):
                # dataset.append({'text':data_texts[i],
                #                 'label':data_labels[i]})
                dataset.append((data_texts[i],
                                data_labels[i]))
        random.shuffle(dataset)
        return dataset
    def __getitem__(self, idx):
        return self.dataset[idx]
    def __len__(self):
        return len(self.dataset)
all_dataset=htl_data()
data_ratio=0.7
train_data_size=int(data_ratio*len(all_dataset))
test_data_size=int(len(all_dataset)-train_data_size)
train_dataset,test_dataset=random_split(
    all_dataset,
    [train_data_size,test_data_size],
    generator=torch.Generator().manual_seed(42)
)
def dataloader_collate_fn(batch):
    texts=[]
    labels=[]
    for data in batch:
        text,label=data
        texts.append(text)
        labels.append(label)
    return texts,torch.tensor(labels,device=device)

train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True,collate_fn=dataloader_collate_fn)
test_dataloader=DataLoader(test_dataset,batch_size=64,shuffle=True,collate_fn=dataloader_collate_fn)
rnn_model=rnn()
optimizer=torch.optim.AdamW(rnn_model.parameters(),lr=2e-4,weight_decay=1e-2)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.7, 
    patience=2,
)

criterion=nn.CrossEntropyLoss()#默认了reduction='mean'

def check_gard(model=rnn_model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient norm: {total_norm}")

def train(epochs=1):
    rnn_model.to(device)
    for epoch in range(epochs):
        rnn_model.train()
        for batch_id,x in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch_text,label=x
            output=[]
            for text in batch_text:
                output.append(rnn_model(text))
            loss=criterion(torch.stack(output),label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), max_norm=0.7)#防止梯度爆炸
            optimizer.step()
            print(f'epoch{epoch+1}:{batch_id}:loss:{loss.item()}')
            train_loss.append(loss.item())
            #print(rnn_model.rnn_cells[0].w_fc.weight)
            #check_gard(rnn_model)
        test_accuracy=0
        test_accurate_num=0
        rnn_model.eval()
        for batch_id,x in enumerate(test_dataloader):
            batch_text,label=x
            for idx,text in enumerate(batch_text):
                output=rnn_model(text)
                test_accurate_num+=(1 if label[idx]==torch.argmax(output).item() else 0)
        test_accuracy=float(test_accurate_num)/float(test_data_size)        
        print(f'{epoch+1}:accyracy:{test_accuracy}')             
train(5)

plt.figure(figsize=(8,6))
plt.scatter(
    range(len(train_loss)),train_loss,
    color='lightblue',linewidth=0.5,alpha=0.8
    )
plt.show()

end=time.time()
run_time=end-start
print(f'time:{int(run_time/3600)}h {int((run_time%3600)/60)}min {int(run_time%60)}s')
# def test():
#     test_word='吃饭'
#     test_vec=w2v_model.in_embed(torch.tensor(vocab[test_word],device=device))
#     print(test_vec)
#     most_similar_word=''
#     most_similar_score=-100000
#     for word,word_num in vocab.items():
#         vec=w2v_model.in_embed(torch.tensor(word_num,device=device))
#         score=torch.sum(vec*test_vec)/(torch.sqrt(torch.sum(vec*vec))
#                                        *
#                                        torch.sqrt(torch.sum(test_vec*test_vec)))
#         if score>most_similar_score and word!=test_word:
#             most_similar_score=score
#             most_similar_word=word
#     print(f'most similar word:{most_similar_word},similar_score:{most_similar_score}')
# print(int('0'))
# test()


