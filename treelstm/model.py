import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import Constants
from transformer.models import Transformer

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, opt):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        '''
        self.ioux = nn.Linear(self.in_dim, self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Wv = nn.Linear(self.mem_dim, self.mem_dim)
        '''
        self.transformer = Transformer(opt)
        #self.W_mv = nn.Parameter(torch.randn(50, 100))
        #self.W_mv_M = nn.Parameter(torch.randn(50, 100))

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs, arcs, S, ttype):
        '''
        num_words = 1
        child_words = []
        residual = []
        residual.append(inputs[tree.idx].unsqueeze(0))
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs, arc, S)
            num_words += tree.children[idx].words
            child_words.append(tree.children[idx].words)
            residual.append(inputs[tree.children[idx].idx].unsqueeze(0))
        
        tree.words = num_words
        child_words.append(tree.words)

        if tree.num_children == 0:
            tree.state = inputs[tree.idx].view(1,-1) #child_h
            tree.words = 1
            return tree.words
        else:
            states = []
            for x in tree.children:
                states.append(x.state)
            child_h = torch.cat(states, dim=0)
        
        x_hat = inputs[tree.idx].view(1,-1)
        tree.state = self.transformer.tree_encode(x_hat, child_h.unsqueeze(0), S, child_words, residual)
        
        return tree.state
        '''
        num_words = 1
        child_words = []
        residual = []
        residual.append(inputs[tree.idx].unsqueeze(0))

        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs, arcs, S, ttype)
            num_words += tree.children[idx].words
            child_words.append(tree.children[idx].words)
            residual.append(inputs[tree.children[idx].idx].unsqueeze(0))
        
        tree.words = num_words
        child_words.append(tree.words)

        if tree.num_children == 0:
            tree.state = inputs[tree.idx].view(1,-1) #child_h
            tree.arc   = arcs[tree.idx].view(1,-1)
            tree.words = 1
            return tree.words
        else:
            states = []
            arc_labels= []
            for x in tree.children:
                states.append(x.state)
                arc_labels.append(x.arc)
            child_h = torch.cat(states, dim=0) #+ self.Wv(torch.cat(arc_labels, dim=0))
            child_arcs = torch.cat(arc_labels, dim=0)
        
        x_hat = inputs[tree.idx].view(1,-1)
        tree.state = self.transformer.tree_encode(x_hat, child_h.unsqueeze(0), child_arcs.unsqueeze(0), S, child_words, residual, ttype)
        tree.arc = arcs[tree.idx].view(1,-1)
        return tree.state


    def forward1(self, tree, inputs, S):
        if tree.num_children == 0:
            tree.state =  inputs[tree.idx].view(1,-1) #child_h
            return [tree.state]
        subtree_list = []
        for idx in range(tree.num_children):
            subtree_list += self.forward1(tree.children[idx], inputs, S)
        dummy = torch.cat(subtree_list, dim=0)
        word_vec = self.transformer.tree_encode1(dummy.unsqueeze(0), S)
        return [word_vec.squeeze(0)]


    def forward_MVRNN(self, tree, inputs, Minputs, S): # for dependency RNNs
        for idx in range(tree.num_children):
            self.forward_MVRNN(tree.children[idx], inputs, Minputs, S)
        
        if tree.num_children == 0:
            tree.Vstate = inputs[tree.idx].view(1,-1) #child_h
            tree.Mstate = Minputs[tree.idx].view(1,50,-1) #child_h
            return 
        else:
            states = []
            matrix = []
            for x in tree.children:
                states.append(x.Vstate.view(1, -1))
                matrix.append(x.Mstate.view(1, 50, -1))
            child_hV = torch.cat(states, dim=0)
            child_hM = torch.cat(matrix, dim=0)
        
        term1 = torch.mm(child_hM[1].view(50,-1), child_hV[0].view(-1,1)).view(1,-1)
        term2 = torch.mm(child_hM[0].view(50,-1), child_hV[1].view(-1,1)).view(1,-1)
        tree.Vstate = torch.tanh(torch.mm(self.W_mv, torch.cat([term1, term2], dim=1).t()).t())
        tree.Mstate = torch.mm( self.W_mv_M, torch.cat([child_hM[0], child_hM[1]], dim=1).t())

        return tree.Vstate.view(1,-1)


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dpout_fc = 0.1
        self.wh = nn.Linear(4 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)
        '''
        self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(4 * self.mem_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.hidden_dim,self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.hidden_dim, self.num_classes),
                )'''

    def forward(self, lvec, rvec):
        lvec = lvec
        rvec = rvec
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)
        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out), dim=1)
        #out = self.classifier(vec_dist)
        return out

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, arc_vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze, opt):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        self.arc_emb = nn.Embedding(arc_vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
            self.arc_emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim, opt)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)
        self.n_positions = 100


    def forward(self, ltree, linputs, rtree, rinputs, larc, rarc):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)

        linputs_arc = self.arc_emb(larc)
        rinputs_arc = self.arc_emb(rarc)
        
        lstate = self.childsumtreelstm(ltree, linputs, linputs_arc, torch.FloatTensor(), 0) 
        rstate = self.childsumtreelstm(rtree, rinputs, rinputs_arc, torch.FloatTensor(), 0)

        lstate1 = self.childsumtreelstm(ltree, linputs, linputs_arc, torch.FloatTensor(), 1) 
        rstate1 = self.childsumtreelstm(rtree, rinputs, rinputs_arc, torch.FloatTensor(), 1)

        output = self.similarity(torch.cat([lstate, lstate1], dim = -1), torch.cat([rstate, rstate1], dim = -1))

        #output = self.similarity(lstate, rstate)
        
        return output
