from __future__ import division
from __future__ import print_function

import os
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# IMPORT CONSTANTS
from treelstm import Constants
# NEURAL NETWORK MODULES/LAYERS
from treelstm import SimilarityTreeLSTM
# DATA HANDLING CLASSES
from treelstm import Vocab
# DATASET CLASS FOR SICK DATASET
from treelstm import SICKDataset
# METRICS CLASS FOR EVALUATION
from treelstm import Metrics
# UTILITY FUNCTIONS
from treelstm import utils
# TRAIN AND TEST HELPER FUNCTIONS
from treelstm import Trainer
# CONFIG PARSER
from config import parse_args

model = None
# MAIN BLOCK
mean = 0
std = 0.225
def main():
    global model
    print("qyx", "mean ", mean, "std ", std)
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    # write unique words from all token files
    sick_vocab_file = os.path.join(args.data, 'sick.vocab')
    if not os.path.isfile(sick_vocab_file):
        token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files = token_files_a + token_files_b
        sick_vocab_file = os.path.join(args.data, 'sick.vocab')
        utils.build_vocab(token_files, sick_vocab_file)

    arc_vocab_file = os.path.join(args.data, 'sick_arc.vocab')
    if not os.path.isfile(arc_vocab_file):
        arc_files_b = [os.path.join(split, 'b.rels') for split in [train_dir, dev_dir, test_dir]]
        arc_files_a = [os.path.join(split, 'a.rels') for split in [train_dir, dev_dir, test_dir]]
        arc_files = arc_files_a + arc_files_b
        arc_vocab_file = os.path.join(args.data, 'sick_arc.vocab')
        utils.build_vocab(arc_files, arc_vocab_file)
        
    
    # get vocab object from vocab file previously written
    vocab = Vocab(filename=sick_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> SICK vocabulary size : %d ' % vocab.size())

    arc_vocab = Vocab(filename=arc_vocab_file,
                  data=None)
    logger.debug('==> SICK ARC vocabulary size : %d ' % arc_vocab.size())

    
    # load SICK dataset splits
    train_file = os.path.join(args.data, 'sick_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SICKDataset(train_dir, vocab, arc_vocab, args.num_classes)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    dev_file = os.path.join(args.data, 'sick_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SICKDataset(dev_dir, vocab, arc_vocab, args.num_classes)
        torch.save(dev_dataset, dev_file)
    logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))
    test_file = os.path.join(args.data, 'sick_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SICKDataset(test_dir, vocab, arc_vocab, args.num_classes)
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))
    

    parser = argparse.ArgumentParser(description='Training Hyperparams')
    # data loading params
    parser.add_argument('-data_path', default = "data")
    
    # network params
    parser.add_argument('-d_model', type=int, default=300)
    parser.add_argument('-d_k', type=int, default=50)
    parser.add_argument('-d_v', type=int, default=50)
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-n_heads', type=int, default=6)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-share_proj_weight', action='store_true')
    parser.add_argument('-share_embs_weight', action='store_true')
    parser.add_argument('-weighted_model', action='store_true') 
    
    # training params
    parser.add_argument('-lr', type=float, default=0.0002)
    parser.add_argument('-max_epochs', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_src_seq_len', type=int, default=300)
    parser.add_argument('-max_tgt_seq_len', type=int, default=300)
    parser.add_argument('-max_grad_norm', type=float, default=None)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-display_freq', type=int, default=100)
    parser.add_argument('-src_vocab_size', type=int, default=vocab.size())
    parser.add_argument('-tgt_vocab_size', type=int, default=vocab.size())
    parser.add_argument('-log', default=None)
    parser.add_argument('-model_path', type=str, default = "")
    
    transformer_opt = parser.parse_args()

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
        vocab.size(),
        arc_vocab.size(),
        args.input_dim,
        args.mem_dim,
        args.hidden_dim,
        args.num_classes,
        args.sparse,
        True,
        transformer_opt)
    
    weight = torch.FloatTensor(args.num_classes).fill_(1)
    criterion = nn.CrossEntropyLoss(weight=weight)
    criterion.size_average = False
    #criterion = nn.KLDivLoss()
    print(model)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sick_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.840B.300d'))
        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    arc_emb_file = os.path.join(args.data, 'sick_arc_embed.pth')
    if os.path.isfile(arc_emb_file):
        arc_emb = torch.load(arc_emb_file)
        print("arc embedding loaded")
    else:
        # load glove embeddings and vocab
        print("creating arc embedding")
        logger.debug('==> ARC vocabulary size: %d ' % arc_vocab.size())
        arc_emb = torch.zeros(arc_vocab.size(), 300, dtype=torch.float, device=device)
        arc_emb.normal_(mean, std)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            arc_emb[idx].zero_()
        torch.save(arc_emb, arc_emb_file)
    


    # plug these into embedding matrix inside model
    matrix_emb_ = torch.zeros(vocab.size(), 2500)
    for i in range(vocab.size()):
        I = torch.eye(50, 50)
        noise = torch.tensor(I.data.new(I.size()).normal_(0, .01))
        I += noise
        matrix_emb_[i] = I.view(-1)

    model.emb.weight.data.copy_(emb)
    model.arc_emb.weight.data.copy_(arc_emb)
    #model.matrix_emb.weight.data.copy_(matrix_emb_)
    #model.pos_emb.weight.requires_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)

    load_checkPoint = torch.load("checkpoints/" + args.expname + ".pt")
    model.load_state_dict(load_checkPoint['model'])
    
    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer, device)

    best = -float('inf')
    for epoch in range(args.epochs):
        
        #train_loss = trainer.train(train_dataset)
        
        #train_loss, train_pred = trainer.test(train_dataset)
        #dev_loss, dev_pred = trainer.test(dev_dataset)
        test_loss, test_pred = trainer.test(test_dataset)
        
        '''
        train_pearson = metrics.pearson(train_pred, train_dataset.labels)
        train_mse = metrics.mse(train_pred, train_dataset.labels)
        logger.info('==> Epoch {}, Train \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, train_loss, train_pearson, train_mse))

        dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
        dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
        logger.info('==> Epoch {}, Dev \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, dev_loss, dev_pearson, dev_mse))
        '''
        #test_loss, test_pred = trainer.test(test_dataset)
        #test_loss = dev_loss
        #test_pred = dev_pred

        test_pearson = metrics.pearson(test_pred, test_dataset.labels)
        test_mse = metrics.mse(test_pred, test_dataset.labels)
        logger.info('==> Epoch {}, Test \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, test_loss, test_pearson, test_mse))
        aaa
        
        if best < dev_pearson:
            best = dev_pearson
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'pearson': test_pearson, 'mse': test_mse,
                'args': args, 'epoch': epoch
            }
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))


if __name__ == "__main__":
    main()
