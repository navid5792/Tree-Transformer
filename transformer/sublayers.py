
import torch
import torch.nn as nn
import torch.nn.init as init

from transformer.modules import Linear
from transformer.modules import ScaledDotProductAttention
from transformer.modules import LayerNormalization


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Parameter((-.5 - .5) * torch.rand(n_heads, d_model, d_k) + .5, requires_grad = True)
        self.w_k = nn.Parameter((-.5 - .5) * torch.rand(n_heads, d_model, d_k) + .5, requires_grad = True)
        self.w_v = nn.Parameter((-.5 - .5) * torch.rand(n_heads, d_model, d_k) + .5, requires_grad = True)

        self.attention = ScaledDotProductAttention(d_k, dropout)

        init.xavier_normal(self.w_q)
        init.xavier_normal(self.w_k)
        init.xavier_normal(self.w_v)

    def forward(self, q, k, v, attn_mask):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
#        print(, "\n\n\n\n\n")
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_v x d_model]
#        print(q_s.size(), k_s.size(), v_s.size())
        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)  # [b_size * n_heads x len_v x d_v]
#        print(q_s.size(), k_s.size(), v_s.size())
#        asas
        # perform attention, result_size = [b_size * n_heads x len_q x d_v]
        outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_heads, 1, 1))

        # return a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.attention = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.proj = Linear(n_heads * d_k, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
#        residual = q
        # outputs: a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask)
#        print(len(outputs), outputs[0].size()) # 8 torch.Size([16, 31, 64])
        # concatenate 'n_heads' multi-head attentions
        outputs = torch.cat(outputs, dim=-1) #torch.Size([16, 31, 512])

        # project back to residual size, result_size = [b_size x len_q x d_model]
#        outputs = self.proj(outputs) # torch.Size([16, 31, 300])

#        outputs = self.dropout(outputs)

        return outputs, attn # layer Norm


class MultiBranchAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout):
        super(MultiBranchAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_branches = n_branches

        self.attention = _MultiHeadAttention(d_k, d_v, d_model, n_branches, dropout)
        # additional weights for BranchedAttention
        self.w_o = nn.Parameter(torch.FloatTensor(n_branches, d_v, d_model)) # 8x64x300
        self.w_kp = torch.rand(n_branches)
        self.w_kp = nn.Parameter(self.w_kp/self.w_kp.sum())
        self.w_a = torch.rand(n_branches)
        self.w_a = nn.Parameter(self.w_a/self.w_a.sum())

        self.pos_ffn = nn.ModuleList([
            PoswiseFeedForwardNet(d_model, d_ff//n_branches, dropout) for _ in range(n_branches)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal(self.w_o)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        d_v, d_model, n_branches = self.d_v, self.d_model, self.n_branches
        residual = q
        b_size = k.size(0)

        # outputs: a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask)
#        print(len(outputs), outputs[0].size()) # 8 torch.Size([16, 31, 64])
#        print(torch.cat(outputs, dim=0).size()) # torch.Size([128, 31, 64])

        outputs = torch.cat(outputs, dim=0).view(n_branches, -1, d_v) # 8, 496, 64
        #outputs = outputs.view(b_size, -1, d_model)
#        print(torch.bmm(outputs, self.w_o).size()) # 8,496,300

        #print(outputs.size(), self.w_o.size())
        #asasas
        
        outputs = torch.bmm(outputs, self.w_o).sum(dim=0).view(b_size, -1, d_model) # 16, 31 ,300


        outputs = self.layer_norm(self.dropout(outputs) + residual) # [b_size x len_q x d_model]
        
#        print(self.w_kp.size(), outputs.size()) # torch.Size([8]) torch.Size([16, 31, 300])

        outputs = [kappa * outputs for kappa in self.w_kp]
#        print(len(outputs)) [8] each has torch.Size([16, 31, 300])
#        for pos_ffn in self.pos_ffn:
#            x = pos_ffn(outputs[0])
#            print(x.size()) # 16x31x300
#            asas

        outputs = torch.cat([pos_ffn(output) for output, pos_ffn \
                      in zip(outputs, self.pos_ffn)], dim=0).view(n_branches, -1, d_model)
    
#        print(outputs.size()) # 128x31x300 reshaped to torch.Size([8, 496, 300])

        outputs = self.w_a.view(-1, 1, 1) * outputs # [n_branches x b_size * len_q x d_model] torch.Size([8, 496, 300])

        outputs = torch.sum(outputs, dim=0).view(b_size, -1, d_model) # [b_size x len_q x d_model] # 1x496x300 --> 16x31x300

        return outputs, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        residual = inputs # inputs: [b_size x len_q x d_model]
        outputs = self.relu(self.conv1(inputs.transpose(1, 2)))
        outputs = self.conv2(outputs).transpose(1, 2) # outputs: [b_size x len_q x d_model]
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs)