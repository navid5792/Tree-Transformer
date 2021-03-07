from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import data_utils
import torch.nn.functional as F
from transformer.modules import Linear
from transformer.modules import PosEncoding
from transformer.layers import EncoderLayer, DecoderLayer, WeightedEncoderLayer, WeightedDecoderLayer
from torch.autograd import Variable
from transformer.sublayers import PoswiseFeedForwardNet
from transformer.sublayers import MultiHeadAttention, MultiBranchAttention
from transformer.modules import ScaledDotProductAttention

def proj_prob_simplex(inputs):
    # project updated weights onto a probability simplex
    # see https://arxiv.org/pdf/1101.6081.pdf
    sorted_inputs, sorted_idx = torch.sort(inputs.view(-1), descending=True)
    dim = len(sorted_inputs)
    for i in reversed(range(dim)):
        t = (sorted_inputs[:i+1].sum() - 1) / (i+1)
        if sorted_inputs[i] > t:
            break
    return torch.clamp(inputs-t, min=0.0)


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(data_utils.PAD).unsqueeze(1)  # b_size x 1 x len_k
    pad_attn_mask = pad_attn_mask.expand(b_size, len_q, len_k) # b_size x len_q x len_k

    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        #print(subsequent_mask)
        #asas
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class tree_encoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False):
        super(tree_encoder, self).__init__()
        self.d_model = d_model
        self.n_layers = 1
        self.layers = nn.ModuleList([MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout) for _ in range(self.n_layers)])
        #self.layers = nn.ModuleList([ScaledDotProductAttention(300, dropout) for _ in range(self.n_layers)])
        #self.layers = nn.ModuleList([MultiBranchAttention(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(self.n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.new_objective = False
        self.proj = nn.Linear(900, 300)
        self.proj1 = nn.Linear(900, 300)
        #self.proj2 = nn.Linear(900, 300)
        
    def forward(self, parent_inputs, mask_parent, child_inputs, mask_child, child_arcs, S, child_words, residual, ttype):
        if ttype == 0:
            q = parent_inputs.reshape(1,300)
            x = child_inputs.reshape(1,-1,300)
            y = child_arcs.reshape(1,-1,300)
            n_facts = x.size(1)
            q = q.unsqueeze(1)
            q = q.repeat(1, n_facts, 1)   
            #print(q.size(), x.size(), y.size(), "----")
            
            #version 0            
            pair_concat = torch.cat((q,y,x), dim=2)
            nn = self.dropout(self.proj(pair_concat.view(-1, 900))).unsqueeze(0)
            residual1 = nn
            self_attn_mask1 = torch.zeros(1, int(residual1.size(1)), int(residual1.size(1))).byte().cuda()
            for layer in self.layers:
                nn, attn = layer(nn, nn, nn, attn_mask=self_attn_mask1)
            nn += residual1
            nn = torch.tanh(torch.sum(nn, dim = 1).view(1, -1))
            outputs1 = nn  
            '''
            pair_concat1 = torch.cat((y,q,x), dim=2)
            nn1 = self.dropout(self.proj1(pair_concat1.view(-1, 900))).unsqueeze(0)
            residual11 = nn1
            self_attn_mask11 = torch.zeros(1, int(residual11.size(1)), int(residual11.size(1))).byte().cuda()
            for layer in self.layers:
                nn1, _ = layer(nn1, nn1, nn1, attn_mask=self_attn_mask11)
            nn1 += residual11
            nn1 = torch.tanh(torch.sum(nn1, dim = 1).view(1, -1))
            outputs2 = nn1 
            '''
            outputs11 = outputs1 #+ outputs2
            print(attn)
            
            
            
            #version 1
            '''
            acx = torch.cat((y, x, q), dim=2).reshape(-1, 1, 900)
            axc = torch.cat((y, q, x), dim=2).reshape(-1, 1, 900)
            pair_concat = torch.cat((acx, axc), 1)
            nn = self.dropout(self.proj1(pair_concat.view(-1, 900))).view(2, -1 , 300)
            nn = torch.sum(nn, dim = 0).unsqueeze(0)
            residual1 = nn
            self_attn_mask1 = torch.zeros(1, int(residual1.size(1)), int(residual1.size(1))).byte().cuda()
            for layer in self.layers:
                nn, _ = layer(nn, nn, nn, attn_mask=self_attn_mask1)
            nn += residual1
            nn = self.dropout(self.proj2(nn.view(-1, 300))).unsqueeze(0)
            nn = torch.tanh(torch.sum(nn, dim = 1).view(1, -1))
            outputs1 = nn
            '''

            #version 2
            '''
            acx = self.dropout(self.proj1(torch.cat((y, x, q), dim=2).reshape(-1, 900))).unsqueeze(0)
            axc = self.dropout(self.proj2(torch.cat((y, q, x), dim=2).reshape(-1, 900))).unsqueeze(0)
            pair_concat = torch.cat((acx, axc), 0)
            nn = torch.sum(pair_concat, dim = 0).unsqueeze(0)
            residual1 = nn
            self_attn_mask1 = torch.zeros(1, int(residual1.size(1)), int(residual1.size(1))).byte().cuda()
            for layer in self.layers:
                nn, _ = layer(nn, nn, nn, attn_mask=self_attn_mask1)
            nn += residual1
            nn = self.dropout(self.proj(nn.view(-1, 300))).unsqueeze(0)
            nn = torch.tanh(torch.sum(nn, dim = 1).view(1, -1))
            outputs1 = nn
            '''

            
            #version 3
            '''
            acx = torch.cat((y, x, q), dim=2).reshape(1, -1, 900)
            axc = torch.cat((y, q, x), dim=2).reshape(1, -1, 900)
            pair_concat = torch.cat((acx, axc), 1)
            nn = pair_concat
            residual1 = nn
            self_attn_mask1 = torch.zeros(1, int(residual1.size(1)), int(residual1.size(1))).byte().cuda()
            for layer in self.layers:
                nn, _ = layer(nn, nn, nn, attn_mask=self_attn_mask1)
            nn += residual1
            nn = self.dropout(self.proj(nn.view(-1, 900))).unsqueeze(0)
            nn = torch.tanh(torch.sum(nn, dim = 1).view(1, -1))
            outputs1 = nn
            '''
            #print("return 0")
            return outputs11.view(1,-1)
        if ttype == 1:
            residual = torch.cat(residual, dim = 0).unsqueeze(0)
            parent_inputs = parent_inputs.reshape(1,300)
            child_inputs = child_inputs.reshape(-1,300)
            child_inputs = torch.cat([parent_inputs, child_inputs], 0)            
            x = []
            for i in range(int(child_inputs.size(0))):
                x.append(child_inputs[i])
            x = torch.cat(x, dim = 0)
            child_inputs = x.view(-1,300)
            hs = child_inputs.unsqueeze(0)
            self_attn_mask = torch.zeros(1, int(hs.size(1)), int(hs.size(1))).byte().cuda()
            n = hs    
            for layer in self.layers:
                n, attn = layer(n, n, n, attn_mask=self_attn_mask)
            n += residual
            mm = torch.tanh(torch.sum(n.squeeze(0),0).unsqueeze(0).unsqueeze(0))
            outputs = mm.squeeze(0)
            #print("return 1")
            print(attn)
            return outputs.view(1,-1)
    
    def forward1(self, hs, S):
        residual = hs
        self_attn_mask = torch.zeros(1, int(hs.size(1)), int(hs.size(1))).byte().cuda()
        n = hs
        for layer in self.layers:
            n, _ = layer(n,n,n, attn_mask=self_attn_mask)
        n += residual
        outputs = torch.tanh(torch.sum(n.squeeze(0),0).unsqueeze(0).unsqueeze(0))
        return outputs

class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.encoder = tree_encoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model, opt.d_ff, opt.n_heads,
                       opt.max_tgt_seq_len, opt.tgt_vocab_size, opt.dropout, opt.weighted_model)

    def trainable_params(self):
        # Avoid updating the position encoding
        params = filter(lambda p: p[1].requires_grad, self.named_parameters())
        # Add a separate parameter group for the weighted_model
        param_groups = []
        base_params = {'params': [], 'type': 'base'}
        weighted_params = {'params': [], 'type': 'weighted'}
        for name, param in params:
            if 'w_kp' in name or 'w_a' in name:
                weighted_params['params'].append(param)
            else:
                base_params['params'].append(param)
        param_groups.append(base_params)
        param_groups.append(weighted_params)
        return param_groups
    
    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
#        enc_freezed_param_ids = set(map(id, self.encoder.pos_emb.parameters()))
#        enc_embeddings = set(map(id, self.encoder.embedding_table.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.pos_emb.parameters()))
#        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        freezed_param_ids = dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def tree_encode(self, x, h, a, S, child_words, residual, ttype):
        parent_inputs = x
        child_inputs = h
        child_arcs = a
        mask_parent = torch.tensor([1]).unsqueeze(0).cuda() 
        if child_inputs.size(1) == 1:
            mask_child = torch.ones(child_inputs.size(0)).expand_as(mask_parent).cuda()
        else:
            mask_child = torch.ones(child_inputs.size(1)).unsqueeze(0).cuda()
        return self.encoder(parent_inputs, mask_parent, child_inputs, mask_child, child_arcs, S, child_words, residual, ttype)

    def tree_encode1(self, dummy, S):
        return self.encoder.forward1(dummy, S)

    def proj_grad(self):
        if self.weighted_model:
            for name, param in self.named_parameters():
                if 'w_kp' in name or 'w_a' in name:
                    param.data = proj_prob_simplex(param.data)
        else:
            pass