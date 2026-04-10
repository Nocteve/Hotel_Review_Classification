def forward(self, x):
    center_word, context_word, n_sample_list = x
    center_vec = self.in_embed(center_word)
    context_vec = self.out_embed(context_word)
    
    # 正样本损失
    pos_loss = -F.logsigmoid(torch.sum(center_vec * context_vec, dim=1))
    
    # 负样本损失
    n_sample_vecs = self.out_embed(n_sample_list)
    n_score_vec = torch.sum(n_sample_vecs * center_vec.unsqueeze(1), dim=2)
    neg_loss = -torch.sum(F.logsigmoid(-n_score_vec), dim=1) / n_sample_list.size(1)
    
    loss = pos_loss + neg_loss
    return loss.mean() if loss.dim() > 0 else loss