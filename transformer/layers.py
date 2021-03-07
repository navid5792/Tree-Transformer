
import torch.nn as nn
from transformer.sublayers import MultiHeadAttention
from transformer.sublayers import MultiBranchAttention
from transformer.sublayers import PoswiseFeedForwardNet


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class WeightedEncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedEncoderLayer, self).__init__()
        self.enc_self_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        return self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, attn_mask=self_attn_mask)


#class DecoderLayer(nn.Module):
#    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
#        super(DecoderLayer, self).__init__()
#        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
#        self.dec_enc_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
#        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
#
#    def forward(self, dec_inputs, enc_inputs, enc_self_attn_mask, dec_self_attn_mask, enc_dec_attn_mask):
#        
#        enc_outputs, enc_self_attn = self.dec_self_attn(enc_inputs, enc_inputs,
#                                                        enc_inputs, attn_mask=enc_self_attn_mask)
#        
#        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
#                                                        dec_inputs, attn_mask=dec_self_attn_mask)
#
#        
#        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
#                                                      enc_outputs, attn_mask=enc_dec_attn_mask)
#        dec_outputs = self.pos_ffn(dec_outputs)
#
#        return dec_outputs, dec_self_attn, dec_enc_attn

#class DecoderLayer(nn.Module):
#    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
#        super(DecoderLayer, self).__init__()
#        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
#        self.dec_enc_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
#        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
#
#    def forward(self, dec_inputs, enc_inputs, enc_self_attn_mask, dec_self_attn_mask, dec_enc_attn_mask, enc_dec_attn_mask,):
#        
#        enc_outputs, enc_self_attn = self.dec_self_attn(enc_inputs, enc_inputs,
#                                                        enc_inputs, attn_mask=enc_self_attn_mask)
#        
#        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
#                                                        dec_inputs, attn_mask=dec_self_attn_mask)
#        
#        enc_out, enc_enc_attn = self.dec_enc_attn(enc_outputs, dec_outputs,
#                                                      dec_outputs, attn_mask=enc_dec_attn_mask)
#        
#        dec_out, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
#                                                      enc_outputs, attn_mask=dec_enc_attn_mask)
#        
#        enc_outputs = self.pos_ffn(enc_out)
#        
#        dec_outputs = self.pos_ffn(dec_out)
#
#        return enc_outputs, dec_outputs
#    
    

class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.child_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.parent_child_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, parent_inputs, child_inputs, parent_self_attn_mask, child_self_attn_mask, child_parent_attn_mask, parent_child_attn_mask):
        

#        enc_outputs, enc_self_attn = self.dec_self_attn(enc_inputs, enc_inputs,
#                                                        enc_inputs, attn_mask=enc_self_attn_mask)
        
#        dec_outputs, dec_self_attn = self.child_self_attn(child_inputs, child_inputs,
#                                                        child_inputs, attn_mask=child_self_attn_mask)
        
#        enc_out, enc_enc_attn = self.dec_enc_attn(enc_outputs, dec_outputs,
#                                                      dec_outputs, attn_mask=enc_dec_attn_mask)
        
        parent_out, parent_child_attn = self.parent_child_attn(parent_inputs, child_inputs,
                                                      child_inputs, attn_mask=parent_child_attn_mask)
        
#        print("Indside decoder : ", parent_out.size(), parent_child_attn.size())
        
        parent_outputs = self.pos_ffn(parent_out)
        
#        dec_outputs = self.pos_ffn(dec_out)

        return parent_outputs


class WeightedDecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedDecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_branches, dropout)
        self.dec_enc_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, attn_mask=enc_attn_mask)
        
        return dec_outputs, dec_self_attn, dec_enc_attn