class Metric:
    def value(self): Float = 0.

    def __str__(self): str = ''


class F1Metric(Metric):
    def __init__(self, precision, recall):
        self.precision = precision
        self.recall = recall

    def value(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def __str__(self):
        return "{:.4g} (precision={:.4g}, recall={:.4g})".format(100*self.value(),
                                                                 100*self.precision,
                                                                 100*self.recall)


class Evaluator:
    def sample(self, predictions, ground_truth):
        pass

    def evaluate(self):
        return 0.

    def clear(self):
        pass


class F1Evaluator(Evaluator):
    def __init__(self):
        self.clear()

    def sample(self, predictions, ground_truth):
        norm_p = (predictions != 0)
        norm_g = (ground_truth != 0)
        tp = (norm_g*norm_p).sum().item()
        tn = ((1 - norm_g)*(1 - norm_p)).sum().item()
        fn = (norm_g*(1 - norm_p)).sum().item()
        fp = ((1 - norm_g)*norm_p).sum().item()
        self.total_tp += tp
        self.total_tn += tn
        self.total_fp += fp
        self.total_fn += fn

    def evaluate(self):
        precision = self.total_tp/(self.total_tp + self.total_fp)
        recall = self.total_tp/(self.total_tp + self.total_fn)
        f1 = 2 * precision * recall / (precision + recall)
        return F1Metric(precision, recall)

    def clear(self):
        self.total_tp = 0
        self.total_tn = 0
        self.total_fp = 0
        self.total_fn = 0
