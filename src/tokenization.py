from vocab import Vocab

import numpy as np


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def tokenize(self, data, data_len=None):
        result = np.zeros((data_len or len(data), self.vocab.max_word_len), dtype=np.int32)
        r_idx = 0
        for sent in data:
            words = Vocab.sep_punctuation(sent).split()
            w_idx = 0
            for word in words:
                token = self.vocab.index[word]
                result[r_idx, w_idx] = token
                w_idx += 1
            r_idx += 1

        return result
