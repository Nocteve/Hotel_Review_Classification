# 修改load_data方法，为每个中心词预生成负样本
for idx, center_word in enumerate(words_num):
    left = max(0, idx - self.window_size)
    right = min(len(words_num) - 1, idx + self.window_size)
    context_indices = list(range(left, right + 1))
    context_indices.remove(idx)  # 移除中心词
    
    # 为所有上下文词生成一次负样本（排除所有上下文词）
    exclude_list = [words_num[i] for i in context_indices]
    n_sample_list = n_sample_generator.get_sample(self.k, exclude_list)
    
    for i in context_indices:
        self.data.append({
            'center_word': center_word,
            'context': words_num[i],
            'n_sample_list': n_sample_list  # 复用相同的负样本
        })