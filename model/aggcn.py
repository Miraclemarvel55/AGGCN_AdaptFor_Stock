"""
GCN model for relation extraction.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import head_to_tree, tree_to_adj
from utils import constant, torch_utils


class GCNClassifier(nn.Module):
	""" A wrapper classifier for GCNRelationModel. """

	def __init__(self, opt, emb_matrix=None):
		super().__init__()
		self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
		in_dim = opt['hidden_dim']
		self.classifier = nn.Linear(in_dim, opt['num_class'])
		self.opt = opt
		self.h = 0

	def forward(self, inputs):
		goals,h = self.gcn_model(inputs)
		self.h = h
		return goals


class GCNRelationModel(nn.Module):
	def __init__(self, opt, emb_matrix=None):
		super().__init__()
		self.opt = opt
		self.emb_matrix = emb_matrix
		self.goal_index = -1

		# create embedding layers
		self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
		self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
		self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
		self.deprel_emb = nn.Embedding(len(constant.DEPREL_TO_ID),opt['deprel_dim'])if opt['deprel_dim']>0 else None
		#一个词可能的情况,|0,1,实体,关系,句子,句内词| = 6
		self.isNoR_emb = nn.Embedding(6,opt['isNoR_dim'])if opt['isNoR_dim']>0 else None
		embeddings = (self.emb, self.pos_emb, self.ner_emb,self.deprel_emb,self.isNoR_emb)
		# self.init_embeddings()

		# gcn layer
		self.gcn = AGGCN(opt, embeddings)

# ------------------------------------------------------------------------------
		# mlp output layer for Relation Matrix 将hidden 输出进行变换升维等,使之适合具体任务.
		self.RM_i_dim = opt['hidden_dim']
		in_dim = opt['hidden_dim']
		layers = [nn.Linear(in_dim, self.RM_i_dim),nn.ELU()]
		for _ in range(self.opt['mlp_layers'] - 1):
			layers += [nn.Linear(self.RM_i_dim, self.RM_i_dim), nn.ReLU()]
		layers += [nn.Linear(self.RM_i_dim, self.RM_i_dim), nn.LeakyReLU()]
		self.mlp_for_RM = nn.Sequential(*layers)

		# to n*n matrix attention like layer for relation Matrix
		self.QueryEmb = nn.Sequential(nn.Linear(self.RM_i_dim, 2*self.RM_i_dim), nn.ELU())
		self.KeyEmb = nn.Sequential(nn.Linear(self.RM_i_dim, 2*self.RM_i_dim), nn.LeakyReLU())
		self.goal_dim2linear2tanh_sequential = nn.Sequential(
			*[nn.Linear(self.RM_i_dim*2,self.RM_i_dim//2),nn.Linear(self.RM_i_dim//2, 1), nn.Tanh()] )
		self.conv2d_dim2dimMatrix = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding=2)
# -----------------------------------------------------------------------------

#******************************************************************************
		# mlp output layer for Query Generating of Relation Distribution
		self.RD_i_dim = opt['hidden_dim']
		self.RD_i_dim_final = 2*self.RD_i_dim
		in_dim = opt['hidden_dim']
		layers = [nn.Linear(in_dim, self.RD_i_dim ), nn.ELU()]
		for _ in range(self.opt['mlp_layers'] - 1) :
			layers += [nn.Linear(self.RD_i_dim, self.RD_i_dim), nn.ReLU()]
		layers += [nn.Linear(self.RD_i_dim, self.RD_i_dim_final), nn.LeakyReLU()]
		self.h_mlp_for_RD_Q = nn.Sequential(*layers)
		
		# mlp output layer for Query Generating of Relation Distribution
		self.RD_i_dim = opt['hidden_dim']
		self.RD_i_dim_final = 2 * self.RD_i_dim
		in_dim = opt['hidden_dim']
		layers = [nn.Linear(in_dim, self.RD_i_dim), nn.ELU()]
		for _ in range(self.opt['mlp_layers'] - 1) :
			layers += [nn.Linear(self.RD_i_dim, self.RD_i_dim), nn.ReLU()]
		layers += [nn.Linear(self.RD_i_dim, self.RD_i_dim_final), nn.LeakyReLU()]
		self.h_mlp_for_RD_K = nn.Sequential(*layers)
		
		# for relation Distribution MM using 将任意两个词组合看作实体对形成查询,任意单个词也都看作关系.
		self.QueryGeneratorAssitorforDeprel = nn.Linear(opt['deprel_dim'], 2*opt['hidden_dim'])
		self.QueryforEntitiesEmb1 = nn.Sequential(nn.Linear(4*opt['hidden_dim'], 2*opt['hidden_dim']), nn.ReLU())
		self.KeyforRelationsEmb = nn.Sequential(nn.Linear(2*opt['hidden_dim'], 2*opt['hidden_dim']), nn.ReLU())

		self.WQueryforDis = nn.Sequential(nn.Linear(2*self.RD_i_dim_final+2*opt['hidden_dim'], 2 * opt['hidden_dim']), nn.ReLU())
		self.WKeyforDis = nn.Sequential(nn.Linear(self.RD_i_dim_final+2*opt['hidden_dim'], 2 * opt['hidden_dim']), nn.ReLU())

#*******************************************************************************
	def init_embeddings(self):
		if self.emb_matrix is None:
			self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
		else:
			self.emb_matrix = torch.from_numpy(self.emb_matrix)
			self.emb.weight.data.copy_(self.emb_matrix)
		# decide finetuning
		if self.opt['topn'] <= 0:
			print("Do not finetune word embedding layer.")
			self.emb.weight.requires_grad = False
		elif self.opt['topn'] < self.opt['vocab_size']:
			print("Finetune top {} word embeddings.".format(self.opt['topn']))
			self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
		else:
			print("Finetune all embeddings.")

	def forward(self, inputs):
		
		h = self.gcn( inputs)
		
		#goal generator
		h_  = h	#[batch_size,stocks,dim]
		h_for_RM = self.mlp_for_RM(h_)	#[batch_size,stocks,dim]
		Q = self.QueryEmb(h_for_RM)
		K = self.KeyEmb(h_for_RM)
		nnMatrix = attention(Q, K,  softmax_needing = True)
		goal_envs_impact_weight = nnMatrix[:,self.goal_index,:].unsqueeze(1)	#[batch_size,1,stocks]

		dim2dimMatrix = attention(Q.transpose(1,2),K.transpose(1,2),softmax_needing=True)
		after_conv_dim2dimM = dim2dimMatrix
		after_conv_dim2dimM = self.conv2d_dim2dimMatrix(after_conv_dim2dimM.unsqueeze(1)).squeeze()
		after_conv_dim2dimM = F.softmax(after_conv_dim2dimM, dim=-2)	#归一化，避免值过大,测试在-2维归一化可以避免发散O(∩_∩)O哈！
		convolutioned_h = torch.cat([h,h_for_RM],2)
		# convolutioned_h = convolutioned_h.bmm(after_conv_dim2dimM)

		goal_dim = goal_envs_impact_weight.bmm(convolutioned_h).squeeze()	#[batch_size,dim]
		goals = self.goal_dim2linear2tanh_sequential(goal_dim)*10

		# print(Q.shape,K.shape,nnMatrix.shape)
		#
		# #结合句法信息直接构建Q K 和最终的 分布三维张量
		# h_RD = (h)
		# h_for_RD = self.h_mlp_for_RD_Q(h_RD)
		# E1Q = h_for_RD.repeat(1,1,maxlen).reshape(*h_for_RD.shape[:2],*h_for_RD.shape[-2:])
		# E2Q = h_for_RD.repeat(1,maxlen,1).reshape(*E1Q.shape)
		# E1E2Q_pre = torch.cat((E1Q,E2Q),-1)
		# E1E2Q_pre = torch.cat((E1E2Q_pre,DependenceMM),-1)
		# E1E2Q = self.WQueryforDis(E1E2Q_pre)
		#
		# h2key_pre = self.h_mlp_for_RD_K(h_RD)
		# h2key_pre = h2key_pre.repeat(1,maxlen,1).reshape(*h2key_pre.shape[:2],*h2key_pre.shape[-2:])
		# h2key_pre = torch.cat((h2key_pre,DependenceMM),-1)
		# KeyforRD = self.WKeyforDis(h2key_pre)
		
		#生成两两实体的可能关系分布,这里的src_mask 可以优化 by sida
		# RelationDistributionMM = attention(E1E2Q,KeyforRD)

		return goals,h

class AGGCN(nn.Module):
	def __init__(self, opt, embeddings):
		super().__init__()
		self.opt = opt
		self.emb, self.pos_emb, self.ner_emb,self.deprel_emb,self.isNoR_emb = embeddings
		self.use_cuda = opt['cuda']
		self.mem_dim = opt['hidden_dim']
		self.interval = 4
		self.d_ = 26 *3	#单个股票单日产生的数据的维度
		self.in_dim = self.d_ * self.interval
		self.stocksOrwords=256

		#1D convolution layer
		self.conv1d = nn.Conv1d(self.stocksOrwords*self.d_,self.stocksOrwords*self.d_,kernel_size=5,stride=1,padding=2)

		# rnn layer
		if self.opt.get('rnn', False):#opt.get('rnn', True) 字典有则取，无则取第二个参数，第二个参数默认是None
			self.input_W_R = nn.Linear(self.stocksOrwords * self.d_, opt['rnn_hidden'])
			self.rnn = nn.LSTM(opt['rnn_hidden'], opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
							   dropout=opt['rnn_dropout'], bidirectional=True)
			##小心哦，因为前面双向lstm 结果维度加倍
			self.input_W_R_reverse = nn.Linear(opt['rnn_hidden']*2,self.stocksOrwords * self.d_)
			self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output
		self.input_W_G = nn.Linear(self.in_dim+self.d_, self.mem_dim)

		self.in_drop = nn.Dropout(opt['input_dropout'])
		self.num_layers = opt['num_layers']

		self.layers = nn.ModuleList()

		self.heads = opt['heads']
		self.sublayer_first = opt['sublayer_first']
		self.sublayer_second = opt['sublayer_second']

		# gcn layer
		for i in range(self.num_layers):
			self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_first, self.heads))
			self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_second, self.heads))

		self.aggregate_W = nn.Linear(len(self.layers) * self.mem_dim, self.mem_dim)

		self.attn = MultiHeadAttention(self.heads, self.mem_dim)

	def encode_with_rnn(self, rnn_inputs, masks, batch_size,cuda = False):
		seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
		h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'],cuda = cuda )
		rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
		rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
		rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
		return rnn_outputs

	def forward(self, inputs):
		embs = inputs #[batch_size,stocks/words,dims/(interval*d_)]

		#一维卷积
		if self.opt.get('cnn',False):
			embs = embs.reshape([*embs.shape[:2],self.interval,self.d_]).transpose(-1,-2)\
					.reshape([embs.shape[0],-1,self.interval])	#[batch_size,stocks/words*d_,interval]
			conv1d_embs = self.conv1d(embs)
			embs = conv1d_embs.reshape([embs.shape[0],-1,self.d_,self.interval]).transpose(-1,-2)\
					.reshape([embs.shape[0],-1,self.d_*self.interval])

		if self.opt.get('rnn', False):
			# 维度变换 抽取出时间维度，因为stocks维度没有时间关系，但是interval 有
			embs = embs.reshape([*embs.shape[:2],self.interval,self.d_])
			embs = embs.transpose(2,1).reshape([embs.shape[0], -1, self.d_ * self.stocksOrwords])
			#到rnn空间的变换映射
			embs = self.input_W_R(embs)
			masks = torch.zeros(embs.shape[:2])
			lstm_embs = self.rnn_drop(self.encode_with_rnn(embs, masks, embs.shape[0],cuda = self.opt['cuda']))
			lstm_embs = self.input_W_R_reverse(lstm_embs)
			#冲lstm_embs 中还原stocks维度
			lstm_embs = lstm_embs.reshape([*lstm_embs.shape[:2],self.stocksOrwords,-1])
			stocks_embs = lstm_embs.transpose(2,1)
			stocks_embs = stocks_embs.reshape([*stocks_embs.shape[:2],-1])
			#终于可以进入gcn了
			gcn_inputs =stocks_embs
		else:
			gcn_inputs = embs
		gcn_inputs = torch.cat([gcn_inputs,inputs[:,:,:self.d_] ],-1)
		#不管走不走rnn，补偿参数，给一个线性变换映射
		gcn_inputs = self.input_W_G(gcn_inputs)
		
		layer_list = []
		outputs = gcn_inputs

		for i in range(len(self.layers)):
			attn_tensor = self.attn(outputs, outputs)
			# print(outputs.shape,attn_tensor.shape);raise
			attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
			outputs = self.layers[i](attn_adj_list, outputs)
			layer_list.append(outputs)

		aggregate_out = torch.cat(layer_list, dim=2)
		dcgcn_output = self.aggregate_W(aggregate_out)

		return dcgcn_output


class GraphConvLayer(nn.Module):
	""" A GCN module operated on dependency graphs. """

	def __init__(self, opt, mem_dim, layers):
		super(GraphConvLayer, self).__init__()
		self.opt = opt
		self.mem_dim = mem_dim
		self.layers = layers
		self.head_dim = self.mem_dim // self.layers
		self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

		# linear transformation
		self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

		# dcgcn block
		self.weight_list = nn.ModuleList()
		for i in range(self.layers):
			self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

		self.weight_list = self.weight_list.cuda() if self.opt['cuda'] else  self.weight_list
		self.linear_output = self.linear_output.cuda() if self.opt['cuda'] else self.linear_output

	def forward(self, adj, gcn_inputs):
		# gcn layer
		denom = adj.sum(2).unsqueeze(2) + 1

		outputs = gcn_inputs
		cache_list = [outputs]
		output_list = []

		for l in range(self.layers):
			# print(adj.shape,outputs.shape);raise
			Ax = adj.bmm(outputs)                     #Ax (< R n*mem_dim
			AxW = self.weight_list[l](Ax)             #AxW (< R n*head_dim
			AxW = AxW + self.weight_list[l](outputs)  # self loop
			AxW = AxW / denom
			gAxW = F.relu(AxW)
			cache_list.append(gAxW)
			outputs = torch.cat(cache_list, dim=2)
			output_list.append(self.gcn_drop(gAxW))

		gcn_outputs = torch.cat(output_list, dim=2)
		gcn_outputs = gcn_outputs + gcn_inputs

		out = self.linear_output(gcn_outputs)

		return out


class MultiGraphConvLayer(nn.Module):
	""" A GCN module operated on dependency graphs. """

	def __init__(self, opt, mem_dim, layers, heads):
		super(MultiGraphConvLayer, self).__init__()
		self.opt = opt
		self.mem_dim = mem_dim
		self.layers = layers
		self.head_dim = self.mem_dim // self.layers
		self.heads = heads
		self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

		# dcgcn layer
		self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
		self.weight_list = nn.ModuleList()

		for i in range(self.heads):
			for j in range(self.layers):
				self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

		self.weight_list = self.weight_list.cuda()  if self.opt['cuda'] else self.weight_list
		self.Linear = self.Linear.cuda()  if self.opt['cuda'] else self.Linear

	def forward(self, adj_list, gcn_inputs):

		multi_head_list = []
		for i in range(self.heads):
			adj = adj_list[i]
			denom = adj.sum(2).unsqueeze(2) + 1
			outputs = gcn_inputs
			cache_list = [outputs]
			output_list = []
			for l in range(self.layers):
				index = i * self.layers + l
				Ax = adj.bmm(outputs)
				AxW = self.weight_list[index](Ax)
				AxW = AxW + self.weight_list[index](outputs)  # self loop
				AxW = AxW / denom
				gAxW = F.relu(AxW)
				cache_list.append(gAxW)
				outputs = torch.cat(cache_list, dim=2)
				output_list.append(self.gcn_drop(gAxW))

			gcn_ouputs = torch.cat(output_list, dim=2)
			gcn_ouputs = gcn_ouputs + gcn_inputs
			multi_head_list.append(gcn_ouputs)

		final_output = torch.cat(multi_head_list, dim=2)
		out = self.Linear(final_output)
		# out (< R n*mem_dim = final_output (< R n*,mem_dim*heads,  *  self.Linear (< R mem_dim*heads ,*men_dim

		return out


def pool(h, mask, type='max'):
	if type == 'max':
		h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
		return torch.max(h, 1)[0]
	elif type == 'avg':
		h = h.masked_fill(mask, 0)
		return h.sum(1) / (mask.size(1) - mask.float().sum(1))
	else:
		h = h.masked_fill(mask, 0)
		return h.sum(1)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True,cuda=False):
	total_layers = num_layers * 2 if bidirectional else num_layers
	state_shape = (total_layers, batch_size, hidden_dim)
	h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
	return (h0.cuda(), c0.cuda() ) if cuda else ( h0,c0)

def attention(query, key, mask=None, dropout=None,distributionPolarFactors=None,softmax_needing=True,masked_fill_value=-1e9):
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, masked_fill_value)
	
	#使用前面生成的两两之间是否有关系来极化分布:exp(ax)
	#这里现在只验证维度变换,详细映射,暂时不做.
	if distributionPolarFactors is not None:
		scores = scores.transpose(-2, -1)
		scores = torch.mul(scores,distributionPolarFactors)
		scores = scores.transpose(-2, -1)
		scores = torch.exp(scores)
	if softmax_needing:
		p_attn = F.softmax(scores, dim=-1)
	else:p_attn = scores
	
	# if distributionPolarFactors is not None:
	#     print(old_scores,scores)
	#     idOfWord = 1
	#     if old_scores[idOfWord].sum() != 0 :
	#         old_p_attn = F.softmax(old_scores,dim = -1)
	#         print(old_p_attn[idOfWord],p_attn[idOfWord])
	#         raise
	if dropout is not None:
		p_attn = dropout(p_attn)
		
	return p_attn


def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

	def __init__(self, h, d_model, dropout=0.1):
		super(MultiHeadAttention, self).__init__()
		assert d_model % h == 0

		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 2)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, mask=None):
		if mask is not None:
			mask = mask.unsqueeze(1)

		nbatches = query.size(0)

		query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
							 for l, x in zip(self.linears, (query, key))]
		attn = attention(query, key, mask=mask, dropout=self.dropout)

		return attn
if __name__ == '__main__':
	opt = constant.default_opt
	aggcnRE = GCNClassifier(opt)

