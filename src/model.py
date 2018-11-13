import torch
from torch.nn import Module, LSTM, Linear, Embedding, Parameter, BCEWithLogitsLoss, Dropout
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

from weight_drop import WeightDrop


class BaselineLSTM(Module):
    def __init__(self, **kwargs):
        super(BaselineLSTM, self).__init__()

        # Instantiate embeddings
        embedding_weights = kwargs.get('embeddings')
        self.hidden_dim = kwargs.get('hidden_dim', 50)
        self.vocab_size = embedding_weights.shape[0]
        self.word_embedding_dim = embedding_weights.shape[1]
        self.embeddings = Embedding(self.vocab_size, self.word_embedding_dim)
        self.embeddings.weight = Parameter(torch.zeros(self.vocab_size, self.word_embedding_dim), requires_grad=False)
        self.embeddings.weight.data = torch.from_numpy(embedding_weights)

        # Initialize LSTM
        lstm_in_dim = self.word_embedding_dim
        layers = kwargs.get("layers", 1)
        self.lstm = LSTM(lstm_in_dim, self.hidden_dim,
                         bidirectional=True,
                         num_layers=layers,
                         batch_first=True)

        # classify from last-out (each direction)
        self.fc1 = Linear(self.hidden_dim*2, 20)
        self.fc2 = Linear(20, 1)

        positive_weight = kwargs.get('positive_weight', 0.5)
        self.pos_weight = (1 - positive_weight)/positive_weight
        self.criterion = BCEWithLogitsLoss()

        self._initialize_biases(self.lstm, is_GRU=False)
        self._initialize_lstm_weights(self.lstm, orthogonal=kwargs.get("orthogonal"))
        self._init_fc_parameters(self.fc1)
        self._init_fc_parameters(self.fc2)

        drop_connect = kwargs.get('drop_connect', .5)
        if drop_connect > 0.:
            base_weights = ['weight_hh_l{}', 'weight_hh_l{}_reverse']
            hh_weights = [w.format(l) for w in base_weights for l in range(layers)]
            self.lstm = WeightDrop(self.lstm,
                                   hh_weights,
                                   drop_connect)
        dropout = kwargs.get('dropout', .5)
        self.dropout = Dropout(dropout, inplace=True)

    #Jozefowicz et al., 2015
    @staticmethod
    def _initialize_biases(rnn, default_bias=0., forget_bias=1., is_GRU=False):
        for names in rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(rnn, name)
                n = bias.size(0)
                bias.data.fill_(default_bias)
                if is_GRU:
                    #Bias vector is of following structure: [b_rg | b_fg | b_gg | b_og]
                    rs_start, rs_end = 0, n//3
                    bias.data[rs_start:rs_end].fill_(forget_bias)
                else:
                    #Bias vector is of following structure: [b_ir | b_iz | b_in]
                    fg_start, fg_end = n//4, n//2
                    bias.data[fg_start:fg_end].fill_(forget_bias)

    #DANGER CHECK THIS
    #Glorot and Bengio, 2010.
    @staticmethod
    def _initialize_lstm_weights(lstm, orthogonal=False, gain=1.):
        for names in lstm._all_weights:
            for name in filter(lambda n: "weight" in n, names):
                weights = getattr(lstm, name)
                if orthogonal and "weight_hh" in name:
                    init.orthogonal_(weights, gain)
                else:
                    init.xavier_uniform_(weights, gain)

    @staticmethod
    def _init_fc_parameters(fc, gain=1., bias=0.):
        init.xavier_uniform_(fc.weight.data, gain)
        fc.bias.data.fill_(bias)

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, input):
        data = input[1]
        if self.is_cuda:
            data = data.cuda()

        # generate mask for lengths
        mask = (data != 0)
        lengths = mask.argmin(dim=1)
        lengths += (lengths == 0).long()*data.size(1)

        # sort on length
        lengths, perm_idx = lengths.sort(0, descending=True)

        # embed
        embedded = self.embeddings(data)

        # process as packed seq
        reordered = embedded[perm_idx, :, :]
        packed = pack_padded_sequence(reordered, lengths, batch_first=True)
        _, rnn_out = self.lstm(packed)
        # concat back and forward end vectors together
        rnn_out_composed = torch.cat([rnn_out[0][0, :, :], rnn_out[0][1, :, :]], dim=1)

        # reorder back to original ordering
        restored_order = torch.empty_like(rnn_out_composed)
        restored_order.scatter_(0, perm_idx.view(-1, 1).expand_as(rnn_out_composed), rnn_out_composed)

        # run through FC for final classifier logit value
        fc1 = self.fc1(self.dropout(restored_order))
        result = self.fc2(fc1)
        return result.squeeze(1)

    def loss(self, predictions, labels):
        if self.is_cuda:
            labels = labels.float().cuda()
        else:
            labels = labels.float()
        self.criterion.weight = (labels > 0.5).float()*(self.pos_weight - 1.0) + 1.0
        return self.criterion(predictions, labels)
