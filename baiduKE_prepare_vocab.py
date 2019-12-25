"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
from collections import Counter

from utils import vocab, constant, helper

# python3 prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('-data_dir', default='dataset/baidu-knowledge-extraction',help='baiduKE directory.')
    parser.add_argument('-vocab_dir', default='dataset/vocab',help='Output vocab directory.')
    parser.add_argument('--wv_dim', type=int, default=300, help='word vector dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train_data.json'
    dev_file = args.data_dir + '/dev_data.json'
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/baiduKE_vocab.pkl'
    emb_file = args.vocab_dir + '/baiduKE_embedding.npy'

    # load files
    print("loading files...")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    tokens = list(set(train_tokens+dev_tokens))
    w2id = {w : i for i, w in enumerate(tokens)}
    
    # load bert vec
    print("building embeddings...max:",len(tokens))
    from bert_serving.client import BertClient
    bc = BertClient(check_length=False)
    embedding = bc.encode(tokens)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(w2id, outfile)
    np.save(emb_file, embedding)
    print("all done.")

def load_tokens(filename):
    tokens = set()
    with open(filename) as infile:
        data = [];import json
        while True:
            try :
                data.append(json.loads(infile.readline()) )
            except:break
        for d in data:
            word_postags = d['postag']
            words, postags = [], []
            [(words.append(word_pos['word']), postags.append(word_pos['pos'])) for word_pos in word_postags if len(word_pos['word'].strip())>0]
            tokens.update(words)
    tokens = list(tokens)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens


if __name__ == '__main__':
    main()


