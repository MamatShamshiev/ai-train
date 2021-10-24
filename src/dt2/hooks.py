import datetime
import logging
import os
import time

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds, log_first_n
from torch.nn import functional as F

from dt2.visualize import visualize_batch_item


class LossEvalHook(HookBase):
    """
    Computes loss on 'data_loader' every 'eval_period' steps (typically used for validation)
    """

    def __init__(self, eval_period, data_loader):
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self.trainer.model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class PredsVisHook(HookBase):
    """
    Visualizes predictions of the model and saves to the storage.
    Args:
        dataset_dicts_train - list of dicts of training samples (dt2 format)
        dataset_dicts_test - list of dicts of test samples (dt2 format)
        metadata - metadata for the datasets
        period - visualize every 'period' steps
        input_format - format of the incoming image
    """

    def __init__(
        self, batches_train, batches_test, metadata, period, input_format="BGR"
    ):
        self.batches_train = batches_train if batches_train else []
        self.batches_test = batches_test if batches_test else []
        self._metadata = metadata
        self._period = period
        self.input_format = input_format
        self.trainer = None

    def before_train(self):
        for i, batch_item in enumerate(self.batches_train):
            out = visualize_batch_item(batch_item, self._metadata, plot=False)
            self.trainer.storage.put_image(
                f"GT_train_#{i+1}", np.transpose(out, (2, 0, 1))
            )

        for i, batch_item in enumerate(self.batches_test):
            out = visualize_batch_item(batch_item, self._metadata, plot=False)
            self.trainer.storage.put_image(
                f"GT_val_#{i+1}", np.transpose(out, (2, 0, 1))
            )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            if self.trainer.model.training:
                training_flag = True
                self.trainer.model.eval()
            else:
                training_flag = False
            for i, batch_item in enumerate(self.batches_train):
                self.visualize_predictions(
                    batch_item, f"pred_train_#{i+1}_iter={next_iter}"
                )
            for i, batch_item in enumerate(self.batches_test):
                self.visualize_predictions(
                    batch_item, f"pred_val_#{i+1}_iter={next_iter}"
                )
            if training_flag:
                self.trainer.model.train()

    def visualize_predictions(self, batch_item, description_string):
        if self.trainer.model.training:
            training_flag = True
            self.trainer.model.eval()
        else:
            training_flag = False
        with torch.no_grad():
            predictions = self.trainer.model.inference([batch_item])
            instances = predictions[0]["instances"]
            sem_seg = predictions[0]["sem_seg"]
            sem_seg = torch.argmax(sem_seg, axis=0)

            image = batch_item["image"]
            image = F.interpolate(
                image[None],
                size=(batch_item["height"], batch_item["width"]),
                mode="nearest",
            )[0]
        preds_item = {
            "image": image,
            "instances": instances,
            "sem_seg": sem_seg,
        }
        out = visualize_batch_item(preds_item, self._metadata, plot=False)
        self.trainer.storage.put_image(description_string, np.transpose(out, (2, 0, 1)))
        if training_flag:
            self.trainer.model.train()


class NumberOfParamsHook(HookBase):
    """
    Logs number of trainable parameters of the model
    """

    def before_train(self):
        num_params = sum(
            p.numel() for p in self.trainer.model.parameters() if p.requires_grad
        )
        log_first_n(
            logging.INFO,
            f"Number of trainable parameters {num_params}",
            name="NumberOfParamsHook",
            key="message",
        )
        self.trainer.storage.put_scalar("number_of_parameters", num_params)


class BestCheckpointerHook(HookBase):
    """
    Saves best model checkpoint.
    Args:
        period (int) - number of iterations to check for metric improvent (typically should be equal to cfg.TEST.EVAL_PERIOD)
        metric_name (str) - name of the metric in trainer storage
        maximize (bool) - maximize or minimize the metric
    TO-DO: save best N checkpoints
    """

    def __init__(
        self,
        period,
        metric_name="bbox/AP50",
        maximize=True,
        checkpoint_name="model_best",
    ):
        self.period = period
        self.metric_name = metric_name
        self.maximize = maximize
        self.metric_best = None
        self.checkpoint_name = checkpoint_name

    def after_step(self):
        if (self.trainer.iter + 1) % self.period == 0:
            metric = self.trainer.storage.latest()[self.metric_name][0]
            is_best = False
            if self.metric_best is None:
                is_best = True
                self.metric_best = metric
            else:
                if self.maximize:
                    if self.metric_best < metric:
                        is_best = True
                        self.metric_best = metric
                else:
                    if self.metric_best > metric:
                        is_best = True
                        self.metric_best = metric
            if is_best:
                with open(
                    os.path.join(
                        self.trainer.cfg.OUTPUT_DIR, f"iter_{self.checkpoint_name}.txt"
                    ),
                    "w",
                ) as f:
                    f.write(str(self.trainer.iter))
                self.trainer.checkpointer.save(self.checkpoint_name)
