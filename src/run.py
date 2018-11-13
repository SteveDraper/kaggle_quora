import getopt
import sys
import gc
import numpy as np
import torchvision
import torch.cuda as cuda

import quora_data
from vocab import Vocab, CanonicalVocab
from tokenization import Tokenizer
from datasets import make_datasets
from embeddings import load_embeddings
from model import BaselineLSTM
from train import Trainer
from evaluator import F1Evaluator


def print_usage():
    print("Usage: python run.py [opts]")
    print("Options:")
    print("  --data=<filepath>           - Path to data file (train or predict as appropriate)")
    print("  --predict                   - Predict on the input set (model must be loaded), default False")
    print("  --output=<filename>         - File to save results to")
    print("  --split=<percentage>        - Percentage to split off for validation and test sets if training (default 5)")
    print("  --hidden_dim=<num>          - LSTM hidden dim (default 50")
    print("  --max_epochs=<num>          - Max epoch count to train for (default 1")
    print("  --batch_size=<num>          - Batch size for training (default 256)")
    print("  --pos_prop=<num>            - Override to the positive class proportion assumption (default empirical on training)")
    print("  --device=<num>              - CUDA device to use (default 0)")
    print("  --help                      - usage help")


def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "d:p",
            ["data=",
             "output=",
             "predict",
             "split=",
             "max_epochs=",
             "hidden_dim=",
             "batch_size=",
             "pos_prop=",
             "device=",
             "help"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)

    # default params
    datafile = None
    output = None
    predict = False
    split = 5
    max_epochs = 1
    device = 0
    hidden_dim = 50
    batch_size = 256
    pos_prop = None

    for o, a in opts:
        if o in ("--data", "-d"):
            datafile = a
        elif o in ("--output"):
            output = a
        elif o in ("--predict", "-p"):
            predict = True
        elif o in ("--split"):
            assert (a < 50) and (a > 0), "Validation/test split must be in (0,50)"
            split = int(a)
        elif o in ("--max_epochs"):
            max_epochs = int(a)
        elif o in ("--hidden_dim"):
            hidden_dim = int(a)
        elif o in ("--batch_size"):
            batch_size = int(a)
        elif o in ("--pos_prop"):
            pos_prop = float(a)
        elif o in ("--device"):
            device = int(a)
        elif o in ("--help"):
            print_usage()
            sys.exit(0)
        else:
            assert False, "unhandled option '{}'".format(str(o))

    assert datafile is not None

    use_cuda = cuda.is_available()
    if use_cuda:
        print("Using CUDA device {}".format(device))
        cuda.set_device(device)

    all_data = quora_data.load(datafile)

    def sentences(records):
        for r in records:
            yield r[1]

    def record_ids(records):
        for r in records:
            yield r[0]

    def record_labels(records):
        for r in records:
            yield int(r[2])

    vocab = Vocab(sentences(all_data))

    print("{} examples".format(len(all_data)))
    print("Vocab size is {}, with max word count of {}".format(len(vocab.index), vocab.max_word_len))

    tokenizer = Tokenizer(vocab)
    tokenized = tokenizer.tokenize(sentences(all_data), data_len=len(all_data))
    ids = np.array(list(record_ids(all_data)), dtype='object')
    if not predict:
        all_labels = list(record_labels(all_data))
        labels = np.array(all_labels, dtype='int8')
        positive_proportion = float(labels.sum())/labels.shape[0]
        print("{}% of the labels are positive".format(100.*positive_proportion))
        if pos_prop is not None:
            positive_proportion = pos_prop
            print("Overriding effective proportion as {}%".format(100.*positive_proportion))
    else:
        labels = None
        positive_proportion = None

    all_data = None  # allow GC
    gc.collect()

    transform = torchvision.transforms.Lambda(quora_data.to_tensors)
    train, validation, test = make_datasets(split, ids, tokenized, labels, transform=transform)

    # Load embeddings
    embedding_files = [
        ('../data/glove.840B.300d/glove.840B.300d.txt', False),
        ('../data/wiki-news-300d-1M/wiki-news-300d-1M.vec', False),
        ('../data/paragram_300_sl999/paragram_300_sl999.txt', False)
    ]
    canonical_vocab = CanonicalVocab(vocab)
    all_embeddings = [load_embeddings(filename, canonical_vocab, binary=binary) for filename, binary in embedding_files]

    for ef, e in zip(embedding_files, all_embeddings):
        print("{} matches {} vocab words".format(ef[0], e[1]))

    # Instantiate the model
    model = BaselineLSTM(embeddings=all_embeddings[0][0],
                         hidden_dim=hidden_dim,
                         positive_weight=positive_proportion)
    if use_cuda:
        model = model.cuda()

    print(model)

    trainer = Trainer(train,
                      validation,
                      test,
                      model=model,
                      evaluator=F1Evaluator(),
                      validate_every=None if max_epochs > 1 else 1000,
                      batch_size=batch_size)
    trainer.train(max_epochs)


if __name__ == "__main__":
    main()
