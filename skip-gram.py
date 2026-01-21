import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
import pandas as pd
import json 
import jieba
import logging
import torch.nn.functional as F
import numpy as np
from vose_sampler import VoseAlias


class negative_sample():
    def __init__(self):
        self.f_path=Path("./f.json")
        self.probs=[]
        self.get_prob()
        self.prob_dict={}
        self.make_alias()
        self.sampler = VoseAlias(self.prob_dict)

    def get_prob(self):
        basic_probs=[]
        with open(self.f_path) as f_json:
            f=json.load(f_json)
            sum=0
            for i in f:
                sum+=i
            for i in f:
                basic_probs.append(i/sum)
        sum=0
        for i in basic_probs:
            sum+=i**(0.75)
        for i in basic_probs:
            self.probs.append(i**(0.75)/sum)
        
    def get_sample_slow_(self,sam_size=1,_except=[]):
        #return[1,2,3,4,5]
        output=[]
        while len(output)<sam_size:
            x=np.random.choice(
                len(self.probs),
                size=1,
                p=self.probs
                )
            if int(x[0]) in _except:
                continue
            else:
                output.append(int(x[0]))
        return output
    def make_alias(self):
        self.prob_dict = {i: p for i, p in enumerate(self.probs)}

    def get_sample(self,sam_size=1,_except=[]):
        output=[]
        while len(output)<sam_size:
            x=self.sampler.sample_n(1)
            if int(x[0]) in _except:
                continue
            else:
                output.append(int(x[0]))
        return output

n_sample_generator=negative_sample() 


class text_dataset(Dataset):
    def __init__(self,window_size=2,k=5):
        super(text_dataset,self).__init__()
        self.data_word_to_num_path=Path('./data.json')
        self.window_size=window_size
        self.data_path=Path("ChnSentiCorp_htl_all.csv")
        self.df=pd.read_csv(self.data_path)
        self.data_texts=self.df['review']
        self.data=[]
        self.VOCAB_SIZE=0
        self.k=k
        self.load_data()
        
    def load_data(self):
        size=0
        word_to_num={}
        with open(self.data_word_to_num_path,'r',encoding='utf-8') as json_file:
            word_to_num=json.load(json_file)
        self.VOCAB_SIZE=len(word_to_num)
        print(f'vocab_size:{len(word_to_num)}')
        for text in self.data_texts:
            size+=1
            if size%500==0:print(f'prepare:{size}')
            #print(type(text))
            if isinstance(text,str):    
                words=jieba.cut(text)
                words_num=[]
                for word in words:
                    words_num.append(word_to_num[word])
                for idx,center_word in enumerate(words_num):
                    left=max(0,idx-self.window_size)
                    right=min(len(words_num)-1,idx+self.window_size)
                    for i in range(left,idx,1):
                        n_sample_list=n_sample_generator.get_sample(self.k,words_num[left:right+1])
                        self.data.append({
                            'center_word':center_word,
                            'context':words_num[i],
                            'n_sample_list':n_sample_list
                            })
                    for i in range(idx,right,1):
                        n_sample_list=n_sample_generator.get_sample(self.k,words_num[left:right+1])
                        self.data.append({
                            'center_word':center_word,
                            'context':words_num[i+1],
                            'n_sample_list':n_sample_list
                            })  
        print('finish loading data')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        center_word=torch.tensor(self.data[index]['center_word'],dtype=torch.long)
        context_word=torch.tensor(self.data[index]['context'],dtype=torch.long)
        n_sample_list=torch.tensor(self.data[index]['n_sample_list'])
        return center_word,context_word,n_sample_list


'''设置一下device'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        # print(n_sample_vecs)
        # print(n_sample_vecs.shape)
        # print(center_vec.unsqueeze(1))
        # print(center_vec.unsqueeze(1).shape)
        n_score_vec=torch.sum(n_sample_vecs*center_vec.unsqueeze(1),dim=2)
        loss-=torch.sum(F.logsigmoid(-n_score_vec))
        #print(n_score_vec.shape)
        #print(loss)
        return loss

model=skip_gram_model()
dataset=text_dataset()
text_loader=DataLoader(dataset,batch_size=128,shuffle=True)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
def train(epochs=1,batch_size=128):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        loss_sum=0

        for batch_id,batch_data in enumerate(text_loader):#用enumerate先接收数据再拆解数据
            # len(batch_data[0])==batch_size ???
            x=[]
            for data in batch_data:
                x.append(data.to(device))
            optimizer.zero_grad()
            loss=model(x)
            loss/=batch_size
            loss.backward()
            optimizer.step()
            if (batch_id+1)%500==0:
                print(f'{epoch+1}:{batch_id+1}:loss:{loss.item()}')
            if (batch_id+1)%10000==0:
                save_path=Path(f'./model/epoch{epoch}/batch{batch_id+1}_weights.pth')
                save_path_root=Path(f'./model/epoch{epoch}')
                if not save_path_root.exists():
                    save_path_root.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),save_path)

train(4)
final_model_path=Path('./model/final.pth')
torch.save(model.state_dict(),final_model_path)

