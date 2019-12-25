import os, time
import jieba
from pyltp import Segmentor, Postagger, Parser, \
    NamedEntityRecognizer, SementicRoleLabeller, SentenceSplitter

from utils.sentence_tools import del_list, slice_by_continue, concate_words_by_index, counter, Singleton, convert_dict, concate_words_by_index


@Singleton
class LtpParser(object):
    """封装哈工大LTP工具包的使用方法"""

    def __init__(self,
                 data_paths: dict,
                 predict_,
                 if_use_jieba=False):
        LTP_DIR = data_paths["LTP_FOLDER"]

        self.if_use_jieba = if_use_jieba

        self.predict_ = predict_

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

    def slice_sentence_and_segmentor(self,
                                     sentence: str,
                                     slice_list: list) -> tuple:
        """
        根据已知NER不仅仅将词语强制合并为一个单词
        并且词语的词性根据LOC/ORG/PER分别进行给出
        """
        sentence_list = list(sentence)
        seg_words, seg_words_index_ = del_list(input_list=sentence_list,
                                               del_slice=[_ for __ in slice_list for _ in __])
        seg_words_index_slice_ = [len(_) for _ in slice_by_continue(seg_words_index_)]
        seg_words_list = list()
        for index, value in enumerate(seg_words_index_slice_):
            pre_number = sum(seg_words_index_slice_[0:index])
            seg_words_list.append(seg_words[pre_number: pre_number + value])
        seg_words = list()
        for _ in seg_words_list:
            seg_words.append("".join(_))
        words = list()
        if self.if_use_jieba:
            for i in seg_words:
                words.extend([_ for _ in jieba.cut(i)])
        else:
            for i in seg_words:
                words.extend([_ for _ in self.segmentor.segment(i)])

        need_slice_list = [(len("".join(words[:index])), len("".join(words[:index])) + \
                            len(word)) for index, word in enumerate(words)]
        need_slice_list = [(i[0], i[1]) for i in need_slice_list]
        seg_words_index = list()
        for i in need_slice_list:
            seg_words_index.append(seg_words_index_[i[0]:i[1]])
        ## 这个实现的功能复杂, 但是后续可能用到, 不删除
        start_index = [min(_) for _ in seg_words_index]
        return (words, start_index)

    def article2sentence(self,
                         article: str) -> list:
        """将article篇章分为sentence"""
        return list(SentenceSplitter.split(article))

    def get_words(self, sentence: str, corpus=None) -> list:
        """将句子分词, 去除掉修饰性质的括号代码等"""

        # 在这里维护一个NE_LIST后续使用语法树做分析的时候更加准确
        # entities , _ = self.predict_(sentence = sentence)
        # entities = entities.get("entities")
        entities = self.predict_(sentence=sentence, corpus=corpus).get("entities")

        slice_list = list()
        if len(entities) > 0:
            # 说明使用NER提取出了对应的命名体识别
            for entity in entities:
                slice_list.append([item for item in range(entity["start"], entity["end"])])
        words, start_index = self.slice_sentence_and_segmentor(sentence=sentence,
                                                               slice_list=slice_list)
        if len(entities) > 0:
            for entity in entities:
                ne_postag = entity.get("type")
                if ne_postag == "PER":
                    ne_postag = "nh"
                elif ne_postag == "LOC":
                    ne_postag = "ns"
                elif ne_postag == "ORG":
                    ne_postag = "ni"
                elif ne_postag == "ZWT":
                    ne_postag = "n"
                elif ne_postag == "PRO":
                    ne_postag = "n"
                # NE_DICT.update({entity["word"] : dict(
                # 	postag = ne_postag, start = entity["start"], end = entity[""]
                # )})
                words.append(entity["word"])
                start_index.append(entity["start"])
        words_dict = dict()
        for index, value in enumerate(start_index):
            words_dict.update({value: words[index]})
        words = list()
        for i in sorted(words_dict.keys()):
            words.append(words_dict.get(i))

        # 在这里直接把使用NER命名体识别的内容返回
        for entity in entities:
            entity_word = entity["word"]
            entity.update({"word_index": words.index(entity_word)})
        return words, entities

    def get_postags(self,
                    words: list,
                    entities: list) -> list:
        """
        根据words和使用predict_得到的NE_List得到对应的postags
        """
        postags = self.postagger.postag(words)
        postags = [_ for _ in postags]
        NE_words = [_["word"] for _ in entities]
        entities_, entities_keys = convert_dict(input_dict_list=entities, input_key="word")
        if len(words) == len(postags):
            for word_index, word_value in enumerate(words):
                if "《" in word_value or "'" in word_value or '"' in word_value:
                    # 如果词语当中有书名号这些, 强制词性为 n
                    postags[word_index] = "n"
                if len(NE_words):
                    if word_value in NE_words:
                        word_type = entities_[entities_keys.index(word_value)].get(word_value).get("type")
                        if word_type == "ORG":
                            postags[word_index] = "ni"
                        elif word_type == "PER":
                            postags[word_index] = "nh"
                        elif word_type == "LOC":
                            postags[word_index] = "ns"
                        elif word_type == "ZWT":
                            # 如果这个词语是职位相关的词语, 那么把这个词语的词性定义为 n名词
                            postags[word_index] = "n"
                        elif word_type == "PRO":
                            postags[word_index] = 'n'
        return postags

    def get_netags(self, words, postags):
        netags = self.recognizer.recognize(words, postags)
        return [item for item in netags]

    def add_art(self,
                words: list,
                postags: list
                ):
        """
        将书名号全部合并, 强制归结词性为名词
        """
        head_index_list, tail_index_list = list(), list()
        for word_index, word_value in enumerate(words):
            if word_value == "《":
                head_index_list.append(word_index)
            elif word_value == "》":
                tail_index_list.append(word_index)
        zipped = list()
        for index, value in enumerate(head_index_list):
            zipped.append([head_index_list[index], tail_index_list[index]])

        for zip in zipped:
            index_start, index_end = zip[0], zip[1]
            article = ""
            for j in range(index_start, index_end + 1):
                article += words[j]
            for k in range(index_start, index_end + 1):
                words[k] = article
            for i in range(index_start, index_end + 1):
                postags[i] = "n"  # 强制把这个书名号的词性转成名词

        for zip in zipped:
            index_start, index_end = zip[0], zip[1]
            for index in range(index_start, index_end):
                try:
                    words.pop(index_start)
                    postags.pop(index_start)
                except:
                    pass
        return words, postags

    def add_quotes(self, words, postags):
        head_index_list, tail_index_list = list(), list()
        for word_index, word_info in enumerate(words):
            if word_info == '“':
                head_index_list.append(word_index)
            elif word_info == '”':
                tail_index_list.append(word_index)
        concate_list = list()
        for index, value in enumerate(head_index_list):
            try:
                concate_list.append([_ for _ in range(head_index_list[index], tail_index_list[index] + 1)])
            except:
                pass
        output_words, output_postags = concate_words_by_index(words=words, postags=postags,
                                                              concate_list=concate_list, flag='n')
        return output_words, output_postags

    def format_labelrole(self, words, postags):
        '''语义角色标注'''
        arcs = self.parser.parse(words, postags)
        roles = self.labeller.label(words, postags, arcs)
        roles_dict = {}
        for role in roles:
            # print("role", role.index,  [(arg.name, arg.range.start, arg.range.end) for arg in role.arguments])
            roles_dict[role.index] = {arg.name: [arg.name, arg.range.start, arg.range.end] for arg in role.arguments}
        return roles_dict

    def build_parse_child_dict(self, words, postags, arcs):
        '''句法分析---为句子中的每个词语维护一个保存句法依存儿子节点的字典'''
        child_dict_list = []
        format_parse_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index].head == index + 1:  # arcs的索引从1开始
                    if arcs[arc_index].relation in child_dict:
                        child_dict[arcs[arc_index].relation].append(arc_index)
                    else:
                        child_dict[arcs[arc_index].relation] = []
                        child_dict[arcs[arc_index].relation].append(arc_index)
            child_dict_list.append(child_dict)
        rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
        relation = [arc.relation for arc in arcs]  # 提取依存关系
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
        for i in range(len(words)):
            # ['ATT', '李克强', 0, 'nh', '总理', 1, 'n']
            try:
                a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i] - 1, postags[rely_id[i] - 1]]
            except Exception as e:
                a = None
            format_parse_list.append(a)
        return child_dict_list, format_parse_list

    def concate_word_by_tag(self,
                            words: list,
                            postags: list,
                            entities: list) -> list:
        """通过顿号 和 字 进行合并对应的词语"""
        target_index_list = list()
        flags_list = list()

        for word_index, word_value in enumerate(words):
            if (word_value == "、" or word_value == "和" or word_value == "与" or word_value == "兼"):
                # and words[word_index-1] not in NE_DICT.keys() and words[word_index+1] not in NE_DICT.keys():
                """出现这种词语时候合并前后两个词语"""
                words[word_index] = '&'
                target_index_list.append(word_index)
                flags_list.append(
                    self.get_postags(words=[words[word_index + 1], words[word_index - 1]], entities=entities))
        flag = counter(input_list=flags_list)
        if len(target_index_list) > 0:
            for index in target_index_list:
                concate_index_set = set() | {index - 1, index, index + 1}
            # 先判断concate_index_list里面是不是连续的, 连续的分为一组, 不连续的分为另外一组
            concate_index_list = list(concate_index_set)
            slice_index_list = slice_by_continue(input_list=concate_index_list)
            # 这里有个BUG 上海工商局和上海海事局联合和开发了一套产品。 这样的会把 前面所有的合并到了一起
            words, postags = concate_words_by_index(words=words,
                                                    postags=postags,
                                                    concate_list=slice_index_list,
                                                    flag=flag)
        return words, postags

    def parser_main(self, input, if_show=0,corpus=None):
        if isinstance(input, list):
            # 说明传过来是个句子
            input = "".join(input)
        words, entities = self.get_words(sentence=input,corpus=corpus)
        words = [_ for _ in words]
        postags = self.get_postags(words=words, entities=entities)
        # 直接得到words对应的词性列表
        postags = [postag for postag in postags]
        # 1. 对书名号相关的合并
        words, postags = self.add_art(words=words, postags=postags)
        # 2. 对双引号相关的合并
        words, postags = self.add_quotes(words=words, postags=postags)
        # 3. 对顿号和 "和" 字这样的合并
        words, postags = self.concate_word_by_tag(words=words, postags=postags, entities=entities)
        # 4.获取语法树相应结构
        arcs = self.parser.parse(words, postags)
        # 5.为每一个词语维护一个儿子节点
        child_dict_list, format_parse_list = self.build_parse_child_dict(words, postags, arcs)
        # 为句子中的每个词语维护一个保存句法依存儿子节点的字典
        roles_dict = self.format_labelrole(words, postags)

        if if_show:
            # 是否展示所生成的结果, 调试代码使用
            print("===========打印===========")
            print("words", [(_, __) for _, __ in enumerate(words)], len(words))  # 对应的分词结果
            print("postags", [(_, __) for _, __ in enumerate(postags)], len(postags))  # 对应的词性标注结果
            print("child_dict_list", [(_, __) for _, __ in enumerate(child_dict_list)],
                  len(child_dict_list))  # 对应的孩子节点的结果
            print("child_dict_list_", child_dict_list, len(child_dict_list))  # 对应的孩子节点的结果
            print("roles_dict", roles_dict, len(roles_dict))  # 对应的语义结果
            print("format_parse_list", format_parse_list, len(format_parse_list))  # 暂时不知道怎么使用
            print("arcs", [(_, (arc.head, arc.relation)) for _, arc in enumerate(arcs)], len(arcs))
            print("NE_list", entities)
            print("===========打印===========")

        return words, postags, arcs, child_dict_list, roles_dict, format_parse_list, entities
if __name__ == '__main__':
    data_paths = dict()
    data_paths["LTP_FOLDER"] = '/media/liuyang/0881ca71-5b07-4293-b65e-68a33f350a5f/home/sida/Tools/pyltp/ltp_data_v3.4.0'
    ltpPS = LtpParser(data_paths=data_paths,predict_ = ner_predict)
    ltpPS.parser_main(input='我是一颗小小的石头.', if_show=True,corpus=None)