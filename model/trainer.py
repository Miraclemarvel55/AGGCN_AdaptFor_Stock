"""
A trainer class.
"""

import torch,math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
from model.aggcn import GCNClassifier
from utils import torch_utils


class Trainer(object):
	def __init__(self, opt, emb_matrix=None):
		raise NotImplementedError

	def update(self, batch):
		raise NotImplementedError

	def predict(self, batch):
		raise NotImplementedError

	def update_lr(self, new_lr):
		torch_utils.change_lr(self.optimizer, new_lr)

	def load(self, filename,new_opt = None):
		try:
			checkpoint = torch.load(filename)
		except BaseException:
			print("Cannot load model from {}".format(filename))
			exit()
		self.model.load_state_dict(checkpoint['model'])
		self.opt = checkpoint['config']
		#try to load with new opt
		# if new_opt is not None and new_opt['cuda']:self.model = self.model.cuda()
		# self.opt = checkpoint['config'] if new_opt is None else new_opt

	def save(self, filename, epoch,best_dev_score):
		self.opt['best_dev_score']=best_dev_score
		params = {
				'model': self.model.state_dict(),
				'config': self.opt,
				}
		try:
			torch.save(params, filename)
			print("model saved to {}".format(filename))
		except BaseException:
			print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
	batch = [torch.from_numpy( data_goal ).float() for data_goal in batch ]
	
	if cuda:
		inputs = Variable(batch[0].cuda())
		goals = Variable( batch[-1].cuda() )
	else:
		inputs = Variable(batch[0])
		goals = Variable(batch[-1])
	
	return inputs,goals

class GCNTrainer(Trainer):
	def __init__(self, opt, emb_matrix=None):
		self.opt = opt
		self.emb_matrix = emb_matrix
		self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
		self.criterion = nn.CrossEntropyLoss()
		self.parameters = [p for p in self.model.parameters() if p.requires_grad]
		if opt['cuda']:
			self.model.cuda()
			self.criterion.cuda()
		self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

	def update(self, batch):
		inputs,goals = unpack_batch(batch, self.opt['cuda'])

		# step forward
		self.model.train()
		self.optimizer.zero_grad()
		predict_goals = self.model(inputs).squeeze()

		Directions = predict_goals * goals
		UnCoDirections = Directions < 0

		loss_fn_RM = torch.nn.MSELoss()
		# loss_fn_RD = torch.nn.BCELoss()
		lossRM = loss_fn_RM(predict_goals,goals)
		loss = lossRM

		if UnCoDirections.sum() > 0:
			loss_UC = (predict_goals-goals)**2 *UnCoDirections.float()
			loss_UC = loss_UC.sum()/UnCoDirections.sum()
			loss_UC = loss_UC**0.5
			loss =lossRM + loss_UC
		print(predict_goals, goals,(predict_goals-goals)**2, UnCoDirections,UnCoDirections.sum(),lossRM,loss,sep='\n')

		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
		self.optimizer.step()
		
		return loss.item()

	def predict(self, batch,result_needing=False):
		self.model.eval()
		inputs,goals = unpack_batch(batch, self.opt['cuda'])
		predict_goals = self.model(inputs).squeeze()
		Directions = predict_goals * goals
		CoDirections = Directions >0
		CoDirections_Accurate = sum(CoDirections).float()/CoDirections.flatten().shape[0]
		
		actions = (predict_goals>0).float()
		profit_loss_s = ( torch.prod( actions*goals/100+1 ) - 1 ) *100
		
		cover=RD_result=0
		
		res = {'CoDirections_Accurate': CoDirections_Accurate, 'profit_loss_s': profit_loss_s,
				 'cover': cover,'RD_result':RD_result }
		
		return res
	
if __name__ == '__main__':
	from utils import constant, torch_utils
	gcnTrainer = GCNTrainer(constant.default_opt)


