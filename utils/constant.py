"""
Define constants.
"""
EMB_INIT_RANGE = 1.0
MAX_LEN = 100
# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

#哈工大pyltp
POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'a':2,'b':3,'c':4,'d':5,'e':6,'g':7,'h':8,'i':9,'j':10,'k':11,'m':12,'n':13,'nd':14,'nh':15,\
            'ni':16,'nl':17,'ns':18,'nt':19,'nz':20,'o':21,'p':22,'q':23,'r':24,'u':25,'v':26,'wp':27,'ws':28,'x':29,'z':30 }
#百度
# POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'n':2,'nr':3,'nz':4,'a':5,'m':6,'c':7,'f':8,'ns':9,'v':10,'ad':11,'q':12,'u':13,'s':14,'nt':15,\
#                 'vd':16,'an':17,'r':18,'xc':19,'t':20,'nw':21,'vn':22,'d':23,'p':24,'w':25,}
DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'SBV':2,'VOB':3,'IOB':4,'FOB':5,'DBL':6,'ATT':7,'ADV':8,'CMP':9,'COO':10,'POB':11,'LAD':12,'RAD':13,'IS':14,'HED':15 }
NEGATIVE_LABEL = 'no_relation'

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}

INFINITY_NUMBER = 1e12

import torch
cuda = torch.cuda.is_available()
default_opt = opt = dict()
opt['hidden_dim'] = 256
opt['num_class']  = 8
opt['vocab_size'] = int(3e4)
opt['topn'] = opt['vocab_size']             #希望精调的前n个 embedding
opt['emb_dim'] = 512
opt['pos_dim'] = 32
opt['ner_dim'] = 16
opt['rnn_layers'] = 4
opt['rnn_dropout'] = 0.16
opt['rnn_hidden'] = 256
opt['input_dropout'] = 0.32
opt['num_layers'] = 4
opt['heads'] = 8
opt['sublayer_first'] = 1
opt['sublayer_second'] = 2
opt['cuda'] = cuda
opt['gcn_dropout'] = 0.32
opt['mlp_layers'] = 2
opt['optim'] = 'sgd'
opt['lr'] = 0.7
opt['num_epoch'] = 100
opt['pooling'] = 'max'
opt['pooling_l2'] = 0.002
