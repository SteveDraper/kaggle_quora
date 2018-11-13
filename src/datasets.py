from torch.utils.data import Dataset


class SplitDataset(Dataset):
    def __init__(self, start, num, *records, transform=None):
        super(SplitDataset, self).__init__()
        self.start = start
        self.num = num
        self.records = records
        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        if item >= self.num:
            raise IndexError
        t = tuple([r[item + self.start] for r in self.records])
        return t if self.transform is None else self.transform(t)


def make_datasets(split, ids, data, labels, transform=None):
    num_records = len(ids)
    num_test = int(num_records*split/100)
    num_validation = num_test
    num_train = num_records - num_test - num_validation

    test_split = SplitDataset(0, num_test, ids, data, labels, transform=transform)
    validation_split = SplitDataset(num_test, num_validation, ids, data, labels, transform=transform)
    train_split = SplitDataset(num_test + num_validation, num_train, ids, data, labels, transform=transform)

    return train_split, validation_split, test_split
