# coding=utf-8
import os
import logging
import re
from logging import handlers

from collections import Counter

def counter(input_list, need_counter = 1):
	"""根据input_list当中的出现的数量将出现次数最多的返回"""
	input_list = [_ for __ in input_list for _ in __]
	if len(input_list)>0:
		return Counter(input_list).most_common(need_counter)[0][0]
	else:
		return None



def del_list(input_list: list,
			 del_slice: list) -> tuple :
	"""
	从list里面删除keep_slice位置的点, 保留其他位置的点
	不仅仅要返回切割后的input_list, 还要返回对应的位置的index信息
	"""
	del_slice = set(del_slice)
	new_list, new_list_index = list(), list()
	for index, value in enumerate(input_list) :
		if index not in del_slice :
			new_list.append(input_list[index])
			new_list_index.append(index)
	return new_list, new_list_index


def keep_list(input_list: list, keep_slice: list) -> tuple :
	"""从list里面保留keep_slice位置的点, 删除其他点的位置"""
	keep_slice = set(keep_slice)
	new_list, new_list_index = list(), list()
	for index, value in enumerate(input_list) :
		if index in keep_slice :
			new_list.append(input_list[index])
			new_list_index.append(index)
	return new_list, new_list_index


def concate_word_by_index(words: list,
						  index_list: list,
						  if_show = False) -> str :
	"""
	通过index_list将words对应的位置合并成为一个短语
	"""
	words, _ = keep_list(input_list = words, keep_slice = index_list)
	if if_show :
		print("合并前的words", words)
		print("合并的index_list", index_list)
		print("合并成为的短语", "".join(words))
	return "".join(words)

def convert_dict(input_dict_list:dict, input_key:str) -> list:
	"""
	输入列表中包含字典, 通过input_key将这个列表中的字典重组
	"""
	output_dict_list = list()
	for i in input_dict_list:
		key_word = i[input_key]
		# i.pop(input_key)
		output_dict_list.append({key_word : i})
	keys = list()
	for i in output_dict_list:
		keys.extend(list(i.keys()))
	return output_dict_list, keys



def slice_by_continue(input_list: list) -> list :
	"""按照input_list里面是否连续进行切分"""
	assert isinstance(input_list, list), "input_list is not a list obj"
	input_list = sorted(input_list)
	want_list, mark_list = list(), list()
	for index, value in enumerate(input_list[1 :]) :
		if value - 1 != input_list[index] :
			mark_list.append(index + 1)
	mark_list.extend([0, len(input_list)])
	mark_list = sorted(mark_list)
	for index, value in enumerate(mark_list[:-1]) :
		want = input_list[value : min(len(input_list), mark_list[index + 1])]
		want_list.append(want)
	return want_list


def concate_words_by_index(words: list,
						   postags: list,
						   concate_list: list,
						   flag:str) -> tuple :
	"""使用 concate_list 对words列表进行合并操作"""
	word_index = [_ for _ in range(len(words))]
	_concate_list = list(set(word_index) - set([_ for __ in concate_list for _ in __]))
	_concate_list.extend(concate_list)
	value_dict = {}
	for value in _concate_list :
		if isinstance(value, list) :
			key = min(value)
		elif isinstance(value, int) :
			key = value
		value_dict.update({key : value})
	temp = []
	for i in sorted(value_dict) :
		want = value_dict[i]
		if isinstance(want, int) :
			want = [want]
		temp.append(want)
	output_words, output_postags = list(), list()
	for concate in temp :
		concate_word = "".join([words[_] for _ in concate])
		if len(concate) == 1 and concate and postags:
			post_list = [postags[_] for _ in concate]
			if post_list:
				postag = post_list[0]
		else :
			postag = flag
		output_words.append(concate_word)
		output_postags.append(postag)
	return output_words, output_postags


def _check_path(path) :
	""" small util """
	if not os.path.exists(path) :
		os.makedirs(path)

def get_data_paths(data_path:str,
				   LTP_FOLDER:str,
				   jieba_plain_txt:str,
				   bi_LSTM_model_path:str,
				   word2id_pkl:str,
				   log_path:str,
				   log_name:str) -> dict:
	"""
	维护代码所需要的路径
	"""
	## 0.对data_path进行操作
	data_paths = dict()
	_check_path(path = data_path)
	if not data_path.endswith("/") :
		data_path += "/"
	
	## 1.维护LTP_FOLDER
	_check_path(path = data_path + LTP_FOLDER)
	if not LTP_FOLDER.endswith("/"):
		LTP_FOLDER += "/"
	data_paths["LTP_FOLDER"] = data_path + LTP_FOLDER
	
	## 2.维护jieba用户自定义词典的位置
	if not os.path.exists(data_path + jieba_plain_txt):
		print("jieba外部用户词典不存在")
	data_paths["jieba_plain"] = data_path + jieba_plain_txt
	
	## 3.双向LSTM model存放的位置
	if not os.path.exists(data_path + bi_LSTM_model_path):
		print("biLSTM model 不存在")
	data_paths["model_path"] = data_path + bi_LSTM_model_path
	
	## 4.word2id pkl 文件的位置
	data_paths["word2id"] = data_path + word2id_pkl
	
	## 5.维护LOG日志文件的位置
	_check_path(path = data_path + log_path)
	data_paths["log"] = data_path + log_path + log_name
	
	return data_paths


def Singleton(cls) :
	"""维护LtpParser的单例模式"""
	_instance = {}
	def _singleton(*args, **kargs) :
		if cls not in _instance :
			_instance[cls] = cls(*args, **kargs)
		return _instance[cls]
	return _singleton

@Singleton
class Logger(object) :
	"""
	维护整个代码的LOG日志
	"""
	level_relations = {
		'info' : logging.INFO,
	}
	
	def __init__(self,
				 data_paths,
				 level,  # 日志的默认形式为info
				 when = 'D',  # 以天数作为时间单位
				 backCount = 3,  # 日志保留三天
				 fmt = "%(asctime)s\r\n%(message)s",
				 if_show_in_screen = False) :  # 是否展示在屏幕上
		self.level = level
		if level == "info" :
			filename = data_paths["log"]
		self.logger = logging.getLogger(filename)
		format_str = logging.Formatter(fmt)  # 设置日志格式
		self.logger.setLevel(self.level_relations.get("info"))  # 设置日志级别
		
		th = handlers.TimedRotatingFileHandler(filename = filename,
											   when = when,
											   backupCount = backCount,
											   encoding = 'utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
		th.setFormatter(format_str)  # 设置文件里写入的格式
		self.logger.addHandler(th)
		if if_show_in_screen :
			sh = logging.StreamHandler()  # 往屏幕上输出
			sh.setFormatter(format_str)  # 设置屏幕上显示的格式
			self.logger.addHandler(sh)  # 把对象加到logger里



def removal(input_parameter):
	"""
	利用字典 keys 值不能重复, 强悍性能去重
	不管是字符串去重 还是 列表去重, 都能很强悍
	"""
	temp_dict = dict()
	for _ in input_parameter:
		temp_dict[_] = None
	return list(temp_dict.keys())


def find_string_by_value(words: list, find_value_list: list) -> list :
	"""
	对于 pyltp 分词后, 找到对应 find_value 的位置
	"""
	return_list = list()
	words_index_list = [_ for _ in range(len(words))]
	index_list = list()
	for find_value in find_value_list:
		index_list.extend([_ for _, __ in enumerate(words) if __==find_value])
	concate_list = slice_by_continue(list(set(words_index_list) - set(index_list)))
	for i in concate_list:
		return_list.append(concate_word_by_index(words, index_list = i))
	return return_list

