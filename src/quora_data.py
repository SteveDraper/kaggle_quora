import csv
import torch


def load(filename, skip_header=True):
    with open(filename, "r") as f:
        r = csv.reader(f)
        if skip_header:
            next(r, None)
        return list(r)


def to_tensors(record):
    if len(record) == 3:
        id, data, label = record
    else:
        id, data = record
        label = None

    data_tensor = torch.from_numpy(data).long()

    return id, data_tensor, label if label is not None else 0
