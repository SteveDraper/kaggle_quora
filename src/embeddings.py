import numpy as np

from vocab import CanonicalVocab


def load_embeddings(filename: str, v: CanonicalVocab, binary:bool=False):
    vocab_matches = 0
    embeddings = None
    exact_match = None
    print("Loading embeddings from {}".format(filename))
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            comps = line.split()
            if len(comps) < 3:
                continue
            if embeddings is None:
                embeddings = np.zeros((len(v.vocab), len(comps)-1), dtype='float32')
                exact_match = np.zeros(len(v.vocab), dtype=np.bool_)
            if len(comps) != embeddings.shape[1] + 1:
                print("Unexpected embedding beginning {} with incorrect dimensionality in {}".format(line[:20], filename))
            else:
                word = v.canonicalize(comps[0])
                indexes = v.canonical_index.get(word, [])
                if len(indexes) > 0:
                    embedding = np.array([float(c) for c in comps[1:]])
                    for idx in indexes:
                        if not exact_match[idx]:
                            embeddings[idx] = embedding
                            vocab_matches += 1
                            if v.vocab.words[idx] == comps[0]:
                                exact_match[idx] = True
    return embeddings, vocab_matches
