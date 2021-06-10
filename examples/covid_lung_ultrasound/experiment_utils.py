import os
import re
import csv
import logging
from pathlib import Path
import wandb
import torch
from torchvision.io import write_video
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from vector_cv_tools.experimental import checkpointing as vckpts


def save_one_video(loader, save_name, scale_0_1=True):
    vid, _ = next(iter(loader))
    vid = vid[0]
    height, width = vid.shape[1:-1]
    if height % 2 != 0:
        height -= 1
    if width % 2 != 0:
        width -= 1
    vid = vid[:, 0:width, 0:height, :]
    if scale_0_1:
        vid = (vid * 255).to(torch.uint8)
    write_video(save_name, vid, fps=8)


def read_csv(csv_path):
    """Assume that csv file is formatted as:
        name1,label1
        name2,label2
        ...
    """
    with open(csv_path) as to_read:
        csv_reader = csv.reader(to_read)
        ret_list = [(name, int(label)) for name, label in csv_reader]
    return ret_list


def init_or_resume_wandb_run(wandb_id_file_path,
                             project_name=None,
                             entity=None,
                             run_name=None,
                             config=None):
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        resume_id = wandb_id_file_path.read_text()
        wandb.init(project=project_name,
                   name=run_name,
                   resume=resume_id,
                   config=config,
                   entity=entity)
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        run = wandb.init(project=project_name,
                         name=run_name,
                         config=config,
                         entity=entity)
        wandb_id_file_path.write_text(str(run.id))

    return wandb.config


def log_and_print(print_str):
    logging.info(print_str)
    print(print_str)


def create_or_restore_training_state(model,
                                     optimizer,
                                     scheduler,
                                     dir_path="./checkpoints/baseline_model"):
    cur_epoch = vckpts.SavebleNumber(0)
    # default initialization into []
    val_loss_list = vckpts.SaveableList()
    # this queries the rng state from the global namespace
    rng = vckpts.SaveableRNG(666)

    prev_epoch_loss_list = vckpts.SaveableList()

    checkpoint = vckpts.SavableCollection(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cur_epoch=cur_epoch,
        val_loss_list=val_loss_list,
        rng=rng,
        prev_epoch_loss_list=prev_epoch_loss_list,
    )

    manager = vckpts.CheckpointManager(
        checkpoint=checkpoint,
        directory=dir_path,
        max_to_keep=3,
        checkpoint_interval=1200,  # 1200 seconds = 20 mins
        keep_best="max",
    )

    # this call does nothing if there is no checkpoint loaded
    # otherwise it loads from the latest checkpoint in the directory
    # specified above
    manager.load_latest_checkpoint()

    if manager.latest_checkpoint is not None:
        min_loss = (checkpoint.val_loss_list[-1]
                    if len(checkpoint.val_loss_list) > 0 else None)
        print_str = (
            f"Checkpoint that has finished epoch {checkpoint.cur_epoch} with "
            f"val loss {min_loss} is loaded from {manager.latest_checkpoint}")

    else:
        print_str = f"No checkpoints found under {manager.directory}, starting from scratch"

    log_and_print(print_str)
    cur_epoch.add_(1)
    return cur_epoch, model, optimizer, scheduler, val_loss_list, manager, prev_epoch_loss_list


class MetricsCalculator:
    """A class to compute the various metrics we care about
        like confusion matrices etc...

    Because this class computes epoch wide states, there is not
        yet a need to save this in the checkpoints if we
        do epoch level saving
    """

    def reset(self):
        self.batch_losses = []
        self.predictions = []
        self.labels = []
        # number of samples we are keeping track of
        self.num_samples = 0
        self.num_correct = 0

    def __init__(self):
        self.reset()

    def add_batch(self, batch_loss, prediction_tensor, labels):
        """Note that prediction_tensor must be shape of (N, num_classes)
            You must also make sure to get rid of the computation graph
                when passing in the batch_loss
        """
        self.batch_losses.append(batch_loss)

        pred_idx = prediction_tensor.argmax(-1)
        self.num_correct += (pred_idx == labels).sum()
        self.num_samples += prediction_tensor.shape[0]

        self.predictions.extend(pred_idx.cpu().cpu().tolist())
        self.labels.extend(labels.cpu().cpu().tolist())

    def compute_and_reset(self, reset=True, out_file=None):
        """Computes the result and then clears the metrics internal
            state to prepare for the next epoch

        If out_file is not None, write out the results there
        """
        conf_matrix = confusion_matrix(self.labels, self.predictions)
        mean_loss = (sum(self.batch_losses) / len(self.batch_losses)
                     if len(self.batch_losses) > 0 else 0)
        mean_accuracy = (self.num_correct /
                         self.num_samples if self.num_samples > 0 else 0)

        pred, labels = self.predictions, self.labels

        if reset:
            self.reset()

        return mean_loss, mean_accuracy, confusion_matrix, pred, labels


def remap_labels(y_truth, y_pred, label_remap):

    assert len(y_truth) == len(y_pred)
    y_truth_new = []
    y_pred_new = []
    for idx, (y1, y2) in enumerate(zip(y_truth, y_pred)):
        y_truth_new.append(label_remap[y1])
        y_pred_new.append(label_remap[y2])

    return y_truth_new, y_pred_new


def compute_precision(y_truth, y_pred, label_remap=None, pos_label=1):
    if label_remap is not None:
        y_truth, y_pred = remap_labels(y_truth, y_pred, label_remap)

    return precision_score(y_truth,
                           y_pred,
                           pos_label=pos_label,
                           average='micro')


def compute_recall(y_truth, y_pred, label_remap=None, pos_label=1):
    if label_remap is not None:
        y_truth, y_pred = remap_labels(y_truth, y_pred, label_remap)

    return recall_score(y_truth, y_pred, pos_label=pos_label, average='micro')


def compute_f1(y_truth, y_pred, label_remap=None, pos_label=1):
    if label_remap is not None:
        y_truth, y_pred = remap_labels(y_truth, y_pred, label_remap)

    return f1_score(y_truth, y_pred, pos_label=pos_label, average='micro')
