for idx, center_word in enumerate(words_num):
    left = max(0, idx - self.window_size)
    right = min(len(words_num) - 1, idx + self.window_size)
    for i in range(left, right + 1):
        if i == idx:
            continue  # 跳过中心词
        # 只排除当前上下文词作为负样本
        n_sample_list = n_sample_generator.get_sample(self.k, [words_num[i]])
        self.data.append({
            'center_word': center_word,
            'context': words_num[i],
            'n_sample_list': n_sample_list
        })