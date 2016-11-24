import mxnet as mx
import sys
import pickle

import xconfig


def get_gpu_number():
    for i in range(100):
        try:
            mx.nd.zeros((1,), ctx=mx.gpu(i))
        except:
            return i


# Read from doc
def read_content(path, max_read_line=sys.maxsize):
    content = []
    count = 0
    with open(path, encoding='utf-8') as ins:
        while True:
            line = ins.readline()
            if not line:
                break
            count += 1
            if count > max_read_line:
                break
            line = line.strip()
            content.append(line.split(' '))
    return content


def load_vocab(path, special=None):
    """
    Load vocab from file, the 0, 1, 2, 3 should be reserved for pad, <unk>, <s>, </s>
    :param path: the vocab
    :param special:
    :return:
    """
    with open(path, 'rb') as f:
        vocab = pickle.load(f)

    if special:
        if not isinstance(special, dict):
            raise Exception('special words not instance of python dict')
        for word, idx in special.items():
            if len(word) == 0:
                continue
            if word == '\n' or word == ' ':
                continue
            if not word in vocab:
                vocab[word] = idx
    return vocab


def sentence2id(sentence, the_vocab):
    words = list(sentence)
    words = [the_vocab[w] if w in the_vocab else the_vocab[xconfig.unk_word] for w in words if len(w) > 0]
    return words


def word2id(word, the_vocab):
    return the_vocab[word] if word in the_vocab else the_vocab[xconfig.unk_word]
