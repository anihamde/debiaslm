import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions # this package provides a lot of nice abstractions for policy gradients
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors
import spacy
import yaml
from collections import OrderedDict
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

import csv
import time
import argparse

parser = argparse.ArgumentParser(description='training runner')
parser.add_argument('--model_type','-m',type=int,default=0,help='Model type (if individual models: 0 for AttnLSTM, 1 for S2S, 2 for AttnGRU; if interpolating models: 0 for Alpha, 1 for Beta, 2 for Gamma)')
parser.add_argument('--model_file','-mf',type=str,default='../../models/HW3/model.pkl',help='Model save target.')
parser.add_argument('--skip_training','-sk',action='store_true',help='Skip training loop and load model from PKL.')
parser.add_argument('--architecture_file','-y',type=str,default='../../models/HW3/model.yaml',help='YAML file containing specs to build model.')
parser.add_argument('--n_epochs','-e',type=int,default=3,help='set the number of training epochs.')
parser.add_argument('--adadelta','-ada',action='store_true',help='Use Adadelta optimizer')
parser.add_argument('--learning_rate','-lr',type=float,default=0.01,help='set learning rate.')
parser.add_argument('--rho','-r',type=float,default=0.95,help='rho for Adadelta optimizer')
parser.add_argument('--weight_decay','-wd',type=float,default=0.0,help='Weight decay constant for optimizer')
parser.add_argument('--accuracy','-acc',action='store_true',help='Calculate accuracy during training loop.')
parser.add_argument('--frequent_ckpt','-ckpt',action='store_true',help='Save checkpoints every epoch, instead of just at the end.')
parser.add_argument('--save_best','-best',action='store_true',help='Save checkpoint after every epoch iff validation ppl improves.')
parser.add_argument('--attn_type','-at',type=str,default='soft',help='attention type')
parser.add_argument('--clip_constraint','-cc',type=float,default=5.0,help='weight norm clip constraint')
parser.add_argument('--word2vec','-w',action='store_true',help='Raise flag to initialize with word2vec embeddings')
parser.add_argument('--embedding_dims','-ed',type=int,default=300,help='dims for word2vec embeddings')
parser.add_argument('--hidden_depth','-hd',type=int,default=1,help='Number of hidden layers in encoder/decoder')
parser.add_argument('--hidden_size','-hs',type=int,default=500,help='Size of each hidden layer in encoder/decoder')
parser.add_argument('--vocab_layer_size','-vs',type=int,default=500,help='Size of hidden vocab layer transformation')
parser.add_argument('--weight_tying','-wt',action='store_true',help='Raise flag to engage weight tying')
parser.add_argument('--bidirectional','-b',action='store_true',help='Raise to make encoder bidirectional')
parser.add_argument('--LSTM_dropout','-ld',type=float,default=0.0,help='Dropout rate inside encoder/decoder LSTMs')
parser.add_argument('--vocab_layer_dropout','-vd',type=float,default=0.0,help='Dropout rate in vocab layer')
parser.add_argument('--interpolated_model','-i',action='store_true',help="Invoke interpolated model, above architecture defining args suppressed, below args activated.")
parser.add_argument('--saved_parameters','-sp',type=str,nargs='+',help="List of model parameter files (PKLs). Needs to match '--saved_architectures' arg.")
parser.add_argument('--saved_architectures','-sa',type=str,nargs='+',help="List of model architecture files (YAMLs). Needs to match '--saved_parameters' arg.")
parser.add_argument('--alpha_embedding_size','-aes',type=int,default=300,help='size of embedding in Alpha.')
parser.add_argument('--convolutional_featuremap_1','-cf1',type=int,default=200,help='Featuremap density for 3x1 conv.')
parser.add_argument('--convolutional_featuremap_2','-cf2',type=int,default=200,help='Featuremap density for 5x1 conv.')
parser.add_argument('--alpha_dropout','-ad',type=float,default=0.5,help='Dropout for Alpha.')
parser.add_argument('--alpha_linear_size','-als',type=int,default=200,help='Size of hidden fully connected layer in Alpha')
parser.add_argument('--freeze_models','-fz',action='store_true',help='raise flag to freeze ensemble member parameters.')
#parser.add_argument('--use_beta','-beta',action='store_true',help='Use the Beta function from models to interpolate. This arg is poorly integrated, I know.')
args = parser.parse_args()
# You can add MIN_FREQ, MAX_LEN, and BATCH_SIZE as args too

model_type = args.model_type
n_epochs = args.n_epochs
learning_rate = args.learning_rate
attn_type = args.attn_type
clip_constraint = args.clip_constraint
word2vec = args.word2vec
rho = args.rho
weight_decay = args.weight_decay






if word2vec:
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    EN.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) # feel free to alter path
    print("Simple English embeddings size", EN.vocab.vectors.size())
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec'
    DE.vocab.load_vectors(vectors=Vectors('wiki.de.vec', url=url)) # feel free to alter path
    print("German embeddings size", DE.vocab.vectors.size())
