# 考虑使用PyTorch内置RNN
class RNNModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=2, dropout=0.5):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x, lengths):
        # x: [batch, seq_len, input_dim]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        # 取最后一个有效时间步的输出
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        last_output = output[torch.arange(output.size(0)), lengths - 1]
        return self.fc(last_output)