"""
Data loader for Stock np data
"""

import json
import random
import torch
import numpy as np


from utils import constant

class DataLoader(object):
	"""
	Load data from json files, preprocess and prepare batches.
	"""
	def __init__(self, filename, batch_size=8, opt=None, vocab=None, evaluation=False,data_source = 'baidu_KE',data_insert=None,interval=4):
		self.interval = interval
		self.batch_size = batch_size
		self.opt = opt
		self.vocab = vocab
		self.eval = evaluation
		self.cache = {}         #读取和处理缓存,减少GPU等待时间

		with open(filename) as infile :
			data = np.load(filename) #[stocks,days,dim]
		self.raw_data = self.data = data
		print(self.raw_data.shape)

	def preprocess(self, data, vocab, opt,d_func,relation_func):
		""" Preprocess the data and convert to ids. """
		pass

	def gold(self):
		""" Return gold labels as a list. """
		return self.labels

	def __len__(self):
		return self.data.shape[1]-self.interval-self.batch_size+1 #TODO 需要详细结算一下位置再确定。

	def __getitem__(self, key,):
		""" Get a batch with index.
		0代表想要预测第0天，数据需要从第1天开始。
		"""
		if not isinstance(key, int):
			raise TypeError
		if key < 0 or key >= len(self):
			raise IndexError
		
		#读取缓存,加速
		try:
			return self.cache[key]
		
		except:
			print('entering except for get data! by sida in loader .')
		
# 		batch_size = 4
# 		N = 10
# 		predictions =  range(0,N-interval )
# 		start =range(1,N-interval+1)
# 		end = range(1+interval-1 ,N)	
# 		batchs = range(0,N-interval-batch_size+1 )
		
		batchSize_words_dim = []
		start = key+1
		end = start+self.batch_size
		# goal_dim_index = range(6,9*self.interval,9)

		for idx in range(start,end):
			#将每个股票理解为一个单词，整个句子 words代表了全部的信息，by sida
			words = self.data[:,idx:idx+self.interval,:]

			#下面加入的都是为了解决就信息太多的问题
			#将位置信息输入type 0
			# positions = np.zeros(words.shape)
			# positions[:,0,:] = 1
			#将位置信息输入type 1
			# positions = np.zeros([*words.shape[:2],1])
			# positions[:,0,:] = 1
			#将位置信息输入type 2
			# positions = np.stack([np.array(range(self.interval))]*words.shape[0]).reshape([*words.shape[:2],1] )
			# positions = np.concatenate([positions]*words.shape[-1],-1)
			#将位置信息输入type 3
			# positions = np.tile(list(range(words.shape[-1])),self.interval*words.shape[0]).reshape(words.shape)
			# positions = np.sin(positions)
			# 将位置信息输入type 4
			now = words[:,0:1,:].repeat(self.interval,1)
			diff = now-words
			positions = now

			temp = np.zeros([*words.shape[:2], words.shape[-1] + positions.shape[-1]+diff.shape[-1] ])
			temp[:, :, ::3] = words
			temp[:, :, 1::3] = positions
			temp[:, :, 2::3] = diff
			# temp = np.zeros( [*words.shape[:2],words.shape[-1]+positions.shape[-1] ] )
			# temp[:,:,::2]=words
			# temp[:,:,1::2]=positions

			words = temp

			# words = words*positions
			words = words.reshape([self.data.shape[0],-1])
			# words = words[:,goal_dim_index]
			batchSize_words_dim.append(words)
		batchSize_words_dim = np.stack(batchSize_words_dim)
		
		goal_stock_idx = -1
		['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol','amount']
		goal_idx_position = 0
		goal = self.data[goal_stock_idx,start-1:end-1,goal_idx_position]
		
		
		#缓存处理的数据,加速哦 by sida
		batch = (batchSize_words_dim,goal)
		self.cache[key] = batch
		# 释放资源,保持self.data的长度功能 by sida
# 		if len(self.cache) == len(self.data):self.raw_data=self.data=np.ones(len(self.cache));print('已缓存预处理数据,原始资源已释放.by sida.')
		
		return batch

	def __iter__(self):
		for i in range(self.__len__()):
			yield self.__getitem__(i)
			
def baidu_KE_get_words_and_maybe_new_words(d, seg) :
	# 专用于baidu KE 数据集的处理函数
	sentence = d['text']
	words = seg.segment(sentence)
	# 查找在主语\谓语\宾语的新词语
	maybe_new_words = set()
	NORs = ['subject', 'object', 'predicate']
	for spo in d['spo_list'] :
		maybe_new_words.update([spo[NOR] for NOR in NORs])
	return words, maybe_new_words

def baidu_KE_get_relation(words, d) :
	#专用与baiduKE数据集的目标关系矩阵,张量生成 函数
	RelationMatrix = np.zeros((len(words),) * 2)
	RelationDistribution = np.zeros((len(words),) * 3)
	speed_up_p = np.eye(N = len(words))  # 行列对角矩阵,用于下面的加速赋值
	NORs = ['subject', 'object', 'predicate']
	for spo in d['spo_list'] :
		i, j, p = [words.index(spo[NOR]) for NOR in NORs]  # 这里只使用首次出现的地方标记
		RelationMatrix[i][j] = 1  # 主语 宾语 1;反之 -1;其他的默认为 0
		RelationMatrix[j][i] = -1
		RelationDistribution[i][j] = speed_up_p[p]
		RelationDistribution[j][i] = speed_up_p[p]
	relation = (RelationMatrix, RelationDistribution)
	return relation

def random_test_get_words_and_maybe_new_words(d, seg) :
	sentence = d['sentence']
	words = list(seg.segment(sentence))
	# 查找在主语\谓语\宾语的新词语
	maybe_new_words = set(d['maybe_new_words']) if 'may_be_new_words' in d.keys() else set()
	return words, maybe_new_words

def random_test_get_relation(words, d=None) :
	#专用与random_test 数据集的目标关系矩阵,张量生成 函数
	RelationMatrix = np.zeros((len(words),) * 2)
	RelationDistribution = np.zeros((len(words),) * 3)
	relation = (RelationMatrix, RelationDistribution)
	return relation

def combinate_words_from_ner(words,postags,nertags):
	# print(' # 合并命名实体识别的词语 by sida ')
	words = list(words)
	# nertags = list(nertags)
	words_after_ner = []
	in_stack_status = False
	stack = []
	for i, nertag in enumerate(nertags) :
		if in_stack_status :
			stack.append(words[i])
			if 'E-' in nertag or i == len(words)-1:
				words_after_ner.append(''.join(stack))
				stack = [];in_stack_status = False
		else :
			if 'O' in  nertag or 'S-' in nertag or ('B-' in nertag and i==len(words)-1):
				words_after_ner.append(words[i])
			elif 'B-' in nertag :
				in_stack_status = True
				stack.append(words[i])
				
	if in_stack_status  and len(stack) != 0:print('合并命名实体出现异常',words,nertags,sep='\n');raise
	return words_after_ner

def map_to_ids(tokens, vocab):
	ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
	return ids


def get_positions(start_idx, end_idx, length):
	""" Get subj/obj position sequence. """
	return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
			list(range(1, length-end_idx))


def get_long_tensor(tokens_list, batch_size):
	""" Convert list of list of tokens to a padded LongTensor. """
	token_len = max(len(x) for x in tokens_list)
	tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
	for i, s in enumerate(tokens_list):
		tokens[i, :len(s)] = torch.LongTensor(s)
	return tokens
def get_extend_tensor(numpy_list,batch_size):
	#将大于等于2维的张量扩充到batch里面最大的那个
	tensor_len = max(len(x) for x in numpy_list)
	numpy_list = [np.pad(arr,[(0,tensor_len-arr.shape[0])]*len(arr.shape),mode = 'constant' ) for arr in numpy_list ]
	return torch.Tensor(numpy_list)
	


def sort_all(batch, lens):
	""" Sort all fields by descending order of lens, and return the original indices. """
	unsorted_all = [lens] + [range(len(lens))] + list(batch)
	sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
	return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
	""" Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
	return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
			else x for x in tokens]
if __name__ == '__main__':
	opt = {'data_dir':'/home/sida/ProjectSave/SAP_Python_0/resources/Test_stockdata/','batch_size':4,'vocab' :0 ,'evaluation':False }
	dtld = DataLoader(opt['data_dir'] + 'codatas_train.npy', opt['batch_size'], opt, evaluation=False)
	for idx,batch in enumerate(dtld):
		print(idx,batch[0].shape,batch[1].shape)
		print(batch[0],batch[1])
		myfirst = (batch[1])
		break
	
	opt = {'data_dir':'/home/sida/ProjectSave/SAP_Python_0/resources/Test_stockdata/','batch_size':4,'vocab' :0 ,'evaluation':False }
	dtld = DataLoader(opt['data_dir'] + 'codatas_dev.npy', opt['batch_size'], opt, evaluation=False)
	for idx, batch in enumerate(dtld) :
		print(idx,batch[0].shape, batch[1].shape)
		your_last = (batch[0])
		
	print(myfirst,your_last,sep='\n')

	


