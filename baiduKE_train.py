"""
Train a model on TACRED.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data.baidu_KE_loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

isServer = False
base_dir = os.environ['HOME']+'/'
if isServer:
    base_dir = '/home/dataexa/sida/'
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=base_dir+'ProjectData/oie_aggcn/dataset/baidu-knowledge-extraction')
parser.add_argument('--vocab_dir', type=str, default=base_dir+'ProjectData/oie_aggcn/dataset/vocab')
parser.add_argument('--emb_dim', type=int, default=768, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=0 , help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=32, help='POS embedding dimension.')
parser.add_argument('--deprel_dim', type=int, default=32, help='deprel embedding dimension.')
parser.add_argument('--isNoR_dim', type=int, default=0, help='一个词可能的情况,|0,1,实体,关系,句子,句内词| = 6,isNoR embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=300, help='hidden state size.')
parser.add_argument('--num_layers', type=int, default=3, help='Num of AGGCN blocks.')
parser.add_argument('--input_dropout', type=float, default=0, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default =0, help='AGGCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=0, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)

parser.add_argument('--heads', type=int, default=3, help='Num of heads in multi-head attention.')
parser.add_argument('--sublayer_first', type=int, default=2, help='Num of the first sublayers in dcgcn block.')
parser.add_argument('--sublayer_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')

parser.add_argument('--conv_l2', type=float, default=0.000, help='L2-penalty.')
parser.add_argument('--isServer',type=bool, default=isServer )
parser.add_argument('--best_dev_score', type=int, default=0)
parser.add_argument('--random_test', dest='random_test', action='store_true', help='Load pretrained model.')


parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0.002, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=3, help='Number of output mlp layers.')
parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=300, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=0.75, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=25, help='Decay learning rate after this epoch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adamax', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=10000, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default=base_dir+'ProjectSave/oie_aggcn/saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str,default = 'saved_models/00/best_model.pt', help='Filename of the pretrained model.')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)

# load vocab
# vocab_file = opt['vocab_dir'] + '/baiduKE_vocab.pkl'
# vocab = Vocab(vocab_file, load=True)
# opt['vocab_size'] = vocab.size
# emb_file = opt['vocab_dir'] + '/baiduKE_embedding.npy'
# emb_matrix = np.load(emb_file)
# assert emb_matrix.shape[0] == vocab.size
# assert emb_matrix.shape[1] == opt['emb_dim']
opt['vocab_size'] = 1000
emb_matrix = None

# load data
opt.update({'data_dir' : os.environ['HOME']+'/ProjectSave/SAP_Python_0/resources/Test_stockdata/',  'vocab' : 0,
       'evaluation' : False} )
train_batch = DataLoader(opt['data_dir'] + 'codatas_train.npy', opt['batch_size'], opt, evaluation = False)
dev_batch = DataLoader(opt['data_dir'] + 'codatas_dev.npy', opt['batch_size'], opt, evaluation = True)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
# vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\tdev_RM_F1\tdev_RD_F1\tbest_dev_score")

# print model info
helper.print_config(opt)

# model
if not opt['load']:
    trainer = GCNTrainer(opt, emb_matrix=emb_matrix)
else:
    # load pretrained model
    # model_file = opt['model_file']
    model_file = opt['save_dir']+'/00/best_model.pt'
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    # trainer  = GCNTrainer(opt)      #尝试直接使用新的配置可能含有cuda 来运行trainer,by sida
    # trainer.load(model_file,model_opt)
    trainer = GCNTrainer(model_opt)
    trainer.load(model_file)
    
    
data_insert = [{'sentence':"查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，"
                                "智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部",
                        'maybe_new_words':['出生地','出生日期','1989年4月17日']},
                    {'sentence':"《离开》是由张宇谱曲，演唱",'maybe_new_words':['歌手','作曲']},
               {'sentence':"我是一棵大树",'maybe_new_words':[]},
               {'sentence':'南京京九思新能源有限公司于2015年05月15日在南京市江宁区市场监督管理局登记成立','maybe_new_words':["成立日期","2015年05月15日",'南京京九思新能源有限公司']},
               {'sentence':'我是一颗铺在万里长城小小的石头。钢铁侠今天回来了,我要去看看.我的家乡在中国福建省南平市政和县桃洋村.'}
                    ]
if opt['random_test']:
    random_test_data = DataLoader(opt['data_dir'] + '/train_data.json', opt['batch_size'], opt, vocab, evaluation=True,\
                              data_source = 'random_test',data_insert=data_insert)
if opt['random_test']:
    for batch in random_test_data:
        result,RD_result,spo_list_M = trainer.predict(batch,result_needing = True)
        print(spo_list_M.shape,np.sum(spo_list_M))
        for coordinate in np.argwhere(spo_list_M>=0):
            b=coordinate[0]; i = coordinate[1]; j = coordinate[-1]
            print(coordinate,b,i,j)
            print(spo_list_M [b,i,j])
            s = batch[0][b,i]
            p = batch[0][b,spo_list_M[b,i,j] ]
            o = batch[0][b,j]
            print(vocab.id2word[s],vocab.id2word[p] ,vocab.id2word[o],sep = ' ')
    assert 1==0,'force raise to break for test'

try:dev_score_history = [model_opt['best_dev_score']]
except: dev_score_history = [0]
print('loaded or pseudo_loaded model score ',dev_score_history)
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f} \n'
max_steps = len(train_batch) * opt['num_epoch']

# start training
mean_loss = [];hist_dev = []
for epoch in range(1, opt['num_epoch']+1):
    temp_loss=[]
    for i, batch in enumerate(train_batch):

        start_time = time.time()
        global_step += 1
        loss = trainer.update(batch)
        temp_loss.append(loss)
        if global_step % opt['log_step'] == 0 or True:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr),'train batch id:',i,'/',len(train_batch))
    mean_loss.append(np.mean(temp_loss))

    # eval on dev
    if True:
        print("Evaluating on dev set...")
        res_es = []
        dev_score = 0
        for i, batch in enumerate(dev_batch):
            print('dev batch id:',i,'/',len(dev_batch))
            res = trainer.predict(batch)
            res_es.append(res)

        meanCoDirection = np.mean([dic['CoDirections_Accurate'] for dic in res_es])
        hist_dev.append(meanCoDirection)

        print('meanCoDirection_hist:',hist_dev)
        print(hist_dev[-3:])
        print('mean_loss: epoch', epoch, mean_loss)
        print(mean_loss[-3:])

        # save
        model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
        if False:trainer.save(model_file, epoch,dev_score)
        if dev_score > max(dev_score_history)and False:
            copyfile(model_file, model_save_dir + '/best_model.pt')
            print("new best model saved.")
            file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
                .format(epoch, dev_RM_F1, dev_RD_F1, dev_score))
        if epoch % opt['save_epoch'] != 0 :
            os.system('rm '+model_file)
        dev_score_history.append(dev_score)
    
        # # lr schedule
        # if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
        #         opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        #     current_lr *= opt['lr_decay']
        #     trainer.update_lr(current_lr)
        #
        # dev_score_history += [dev_score]
        print("")

print("Training ended with {} epochs.".format(epoch))

