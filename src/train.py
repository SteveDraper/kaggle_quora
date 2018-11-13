import torch
from torch.nn import Module
from torch import set_grad_enabled
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from copy import deepcopy

from evaluator import Evaluator


class Trainer:
    """ Implements logic for training a model against its datasets, including
        monitoring and validation
    """
    def __init__(self,
                 train_ds: Dataset,
                 validation_ds: Dataset,
                 test_ds: Dataset,
                 model: Module,
                 num_loaders=0,
                 batch_size=64,
                 evaluator: Evaluator=None,
                 validate_every=None):
        self.train_ds = train_ds
        self.validation_ds = validation_ds
        self.test_ds = test_ds
        self.model = model
        self.num_loaders = num_loaders
        self.batch_size = batch_size
        self.evaluator = evaluator
        self.validate_every = validate_every
        self.optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad])
        self.max_grad_norm = None
        self.threshold = 0.5
        self.cached_validation_set = None
        self.evaluation_batch_size = 256

    def _log_info(self, str: str):
        print(str)

    def train_epoch(self, epoch: int):
        dataloader = DataLoader(self.train_ds,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_loaders)
        train_loss = 0.
        epoch_train_loss = 0.
        batch_sample_count = 0
        epoch_sample_count = 0
        if self.evaluator is not None:
            self.evaluator.clear()

        for i_batch, sample_batched in enumerate(dataloader):
            batch_size = sample_batched[-1].size(0)

            self.optimizer.zero_grad()
            self.model.train(True)
            y = self.model(sample_batched)

            loss = self.model.loss(y, sample_batched[-1])
            loss.backward()

            # Clip gradients
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            batch_loss = loss.item()*batch_size
            train_loss += batch_loss
            epoch_train_loss += batch_loss
            batch_sample_count += batch_size
            epoch_sample_count += batch_size

            if self.evaluator is not None:
                self.evaluator.sample(y.cpu() > self.threshold, sample_batched[-1])

            if self.validate_every is not None and i_batch % self.validate_every == self.validate_every - 1:
                self._log_info("Epoch {}, batch {}: training loss {}".format(epoch,
                                                                             i_batch,
                                                                             str(train_loss/batch_sample_count)))
                batch_sample_count = 0
                train_loss = 0.

                validation_loss, validation_metric = self.validate()
                self._log_info("Epoch {}, batch {}: validation loss = {}, validation metric {}".format(
                    epoch,
                    i_batch,
                    str(validation_loss),
                    str(validation_metric)
                ))

        self._log_info("Epoch {}: training loss {}".format(epoch, str(epoch_train_loss/epoch_sample_count)))
        if self.evaluator is not None:
            self._log_info("Epoch {} training metric - {}".format(epoch, str(self.evaluator.evaluate())))

        validation_loss, validation_metric = self.validate()
        self._log_info("Epoch {}: validation loss = {}, validation {}".format(
            epoch,
            str(validation_loss),
            str(validation_metric)
        ))

    def train(self, max_epochs):
        self.train_resume(end_epoch=max_epochs)

    def serialize(self):
        return {
            "model": self.model.serialize()
        }

    def deserialize(self, state_dict):
        self.model.deserialize(state_dict['model'])

    def train_resume(self,
                     end_epoch: int):
        epoch = 0
        while epoch < end_epoch:
            self.train_epoch(epoch)
            epoch += 1

    def validate(self, override_model=None, override_sample_batch_count=None):
        if self.cached_validation_set is None:
            self.cached_validation_set = DataLoader(self.cache_dataset(self.validation_ds),
                                                    batch_size=override_sample_batch_count or self.evaluation_batch_size)

        return self._evaluate_on_ds(self.cached_validation_set,
                                    override_model=override_model)

    def test(self, override_model=None):
        dl = DataLoader(self.test_ds, batch_size=self.evaluation_batch_size)
        return self._evaluate_on_ds(dl, override_model=override_model)

    def _evaluate_on_ds(self, dl, override_model=None):
        if self.evaluator is not None:
            evaluator = deepcopy(self.evaluator)
            evaluator.clear()
        else:
            evaluator = None
        model = override_model or self.model
        model.train(False)
        with set_grad_enabled(False):
            sample_count = 0
            loss_total = 0.
            for i_batch, sample_batched in enumerate(dl):
                num_samples = sample_batched[-1].size(0)

                y = model(sample_batched[:-1])

                loss_total += model.loss(y, sample_batched[-1]).item()*num_samples
                sample_count += num_samples

                if evaluator is not None:
                    predictions = (y > self.threshold)
                    evaluator.sample(predictions.cpu(), sample_batched[-1])

                # ensure GC for GPU memory
                del y

            loss = loss_total/sample_count if sample_count > 0 else 0.

            return loss, (evaluator.evaluate() if evaluator else 0.)

    def cache_dataset(self, ds):
        cached = []
        for r in ds:
            cached.append(r)

        return cached
