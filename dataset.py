import pandas as pd 
import jieba 
import json
import logging

df=pd.read_csv('ChnSentiCorp_htl_all.csv')
#print(df.head())
f_word=[]#词频率统计
id={}
flag=0  
labels=df['label']
reviews=df['review']
print(reviews[0])
for text in reviews:
    if isinstance(text, float) or text is None:
        pass
    else:
        words=jieba.cut(text)
    
    for word in words:
        if word in id :
            f_word[id[word]]+=1
        else:
            id[word]=flag
            f_word.append(1)
            flag+=1    
with open('data.json','w',encoding='utf-8') as f:
    json.dump(id,f,ensure_ascii=False,indent=4)
with open('f.json','w',encoding='utf-8') as f:
    json.dump(f_word,f,ensure_ascii=False,indent=2)

# json_string=json.dumps(f_word,ensure_ascii=False,indent=2)
# print(json_string)