for text in reviews:
    if not isinstance(text, str) or pd.isna(text):
        continue
    words = jieba.cut(text)
    for word in words:
        if word in id:
            f_word[id[word]] += 1
        else:
            id[word] = flag
            f_word.append(1)
            flag += 1

# 在skip-gram.py中：
words_num = []
for word in words:
    num = word_to_num.get(word)
    if num is not None:
        words_num.append(num)
    else:
        # 处理未知词，如使用特殊标记或跳过
        words_num.append(0)  # 假设0是UNK标记