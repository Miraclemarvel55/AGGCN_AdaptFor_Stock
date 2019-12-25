import json
import os, time
import jieba
from pyltp import Segmentor, Postagger, Parser,NamedEntityRecognizer, SementicRoleLabeller, SentenceSplitter

class LtpParser(object):
    """封装哈工大LTP工具包的使用方法"""

    def __init__(self,
                 data_paths: dict,
                 if_use_jieba=False):
        LTP_DIR = data_paths["LTP_FOLDER"]

        self.if_use_jieba = if_use_jieba

        self.data_paths = data_paths

        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(LTP_DIR, "cws.model"))

        self.postagger = Postagger()
        self.postagger.load(os.path.join(LTP_DIR, "pos.model"))

        self.parser = Parser()
        self.parser.load(os.path.join(LTP_DIR, "parser.model"))

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(LTP_DIR, "ner.model"))

        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(LTP_DIR, 'pisrl.model'))

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(LTP_DIR, 'ner.model'))

        if if_use_jieba:
            start_time = time.time()
            jieba.load_userdict(data_paths["jieba_plain"])
            end_time = time.time()
            print("加载jieba词典耗费{}".format(end_time - start_time))

    def parser_main(self, input, if_show=0,corpus=None):
        words = self.segmentor.segment(input)
        words = list(words)
        postags = self.postagger.postag(words)
        postags = list(postags)
        nertags = self.recognizer.recognize(words, postags)
        nertags = list(nertags)
        print(words)
        print(postags)
        print(nertags)
        #合并命名实体识别的词语 by sida
        words_after_ner =[];in_stack_status = False;stack=[]
        for i,nertag in enumerate(nertags):
            if in_stack_status:
                if 'E-' in nertag:
                    stack.append(words[i])
                    words_after_ner.append(''.join(stack))
                    stack = [];in_stack_status = False
                else:stack.append(words[i])
            else:
                if nertag in ['O','S']:words_after_ner.append(words[i])
                elif 'B-' in nertag:
                    in_stack_status = True;stack.append(words[i])
                    
        assert in_stack_status==False and len(stack)==0,'合并命名实体出现异常'
        words = words_after_ner
        postags = self.postagger.postag(words)
        postags = list(postags)
        nertags = self.recognizer.recognize(words, postags)
        nertags = list(nertags)
        print(words)
        print(postags)
        print(nertags)
        
        arcs = self.parser.parse(words,postags)  # 句法分析
        rely_id = [arc.head for arc in arcs] # 提取依存父节点id
        print(rely_id)
        relation = [arc.relation for arc in arcs] # 提取依存关系
        heads = ['Root' if id == 0 else words[id-1] for id in rely_id] # 匹配依存父节点词语
        
        if if_show:
            # 是否展示所生成的结果, 调试代码使用
            print("===========打印===========")
            print("words", words)  # 对应的分词结果
            print("postags", postags)  # 对应的词性标注结果
            print("NE_list", nertags)
            for i in range(len(words)):
                print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')' )
            print("===========打印===========")

if __name__ == '__main__':
    data_paths = dict()
    data_paths["LTP_FOLDER"] = '/media/liuyang/0881ca71-5b07-4293-b65e-68a33f350a5f/home/sida/Tools/pyltp/ltp_data_v3.4.0/'
    ltpPS = LtpParser(data_paths=data_paths)
    ltpPS.parser_main(input="南京京九思新能源有限公司于2015年05月15日在南京市江宁区市场监督管理局登记成立.sep.南京京九思新能源有限公司", if_show=True,corpus=None)
    
    filename = 'dev_data.json'
    i = 1
    with open(filename) as f :
        while True :
            i += 1;#print(i)
            record1 = json.loads(f.readline())
            # u'postag': [{
            #                 u'word' : u'\u6b66\u6c49\u571f\u6728\u77f3\u5efa\u7b51\u88c5\u9970\u8bbe\u8ba1\u5de5\u7a0b\u6709\u9650\u516c\u53f8',
            #                 u'pos' : u'nt'}, {u'word' : u'\u4e8e', u'pos' : u'p'},
            #             {u'word' : u'2012\u5e7410\u670825\u65e5', u'pos' : u't'},
            #             {u'word' : u'\u6210\u7acb', u'pos' : u'v'}]
            word_postags = record1['postag']
            
            words,postags =[],[]
            [( words.append(word_pos['word']),postags.append(word_pos['pos']) ) for word_pos in word_postags]
            
            self = ltpPS
            arcs = self.parser.parse(words, postags)  # 句法分析
            rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
            relation = [arc.relation for arc in arcs]  # 提取依存关系
            heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
            assert 1==0
            
            
            

