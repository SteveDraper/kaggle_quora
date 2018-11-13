class Vocab:
    def __init__(self, data, reserve_pad=True):
        self.index = {}
        self.words = []
        if reserve_pad:
            self.words.append("<PAD>")
            self.idx = 1
        else:
            self.idx = 0
        self.max_word_len = 0
        for sent in data:
            words = self.sep_punctuation(sent).split()
            for word in words:
                if word not in self.index:
                    self.index[word] = self.idx
                    self.words.append(word)
                    self.idx += 1
            if len(words) > self.max_word_len:
                self.max_word_len = len(words)
                print("Longest sentence: {}".format(sent))

    def __len__(self):
        return len(self.words)

    @staticmethod
    def sep_punctuation(s):
        result = ''
        in_alpha = True
        for c in s:
            if (not c.isspace()) and (in_alpha != c.isalnum()):
                if (len(result) > 0) and not result[-1].isspace():
                    result = result + ' '
                in_alpha = c.isalnum()
            result = result + c
        # Hack to handle equational forms that are basically markup
        return result if len(result) < 1.2*len(s) else s


class CanonicalVocab:
    def __init__(self, vocab):
        self.vocab = vocab
        self.canonical_index = {}
        for word, token in vocab.index.items():
            c_word = self.canonicalize(word)
            tokens = self.canonical_index.get(c_word, [])
            tokens.append(token)
            self.canonical_index[c_word] = tokens

    def canonicalize(self, word):
        return word.lower()
