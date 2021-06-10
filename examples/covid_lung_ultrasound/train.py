import os
import logging
import argparse
from pathlib import Path
import torch
import wandb
from torch.utils.data import DataLoader, sampler
from vector_cv_tools import utils
from model import all_models, get_model
from covid_dataset import COVIDDataset
from transforms import get_transform, all_transforms
from experiment_utils import (save_one_video, create_or_restore_training_state,
                              log_and_print, MetricsCalculator,
                              init_or_resume_wandb_run, compute_f1,
                              compute_precision, compute_recall)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_args():
    parser = argparse.ArgumentParser(description="COVID Dataset Training")

    # dataset args
    parser.add_argument(
        "--data_dir",
        type=str,
        default='./COVID-US/data/video/')
    parser.add_argument(
        "--split_dir",
        type=str,
        default='./COVID-US/data/splits')
    parser.add_argument("--num_classes", type=int, default=3)

    # experiment args
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project_name",
                        type=str,
                        default="COVID-US-ClassCombined")
    parser.add_argument("--wandb_entity", type=str, default="vector-institute")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--transform",
                        type=str,
                        default="base",
                        choices=all_transforms())
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")

    # training args

    # we only have 5 folds to --fold_num goes from 0 to 4
    parser.add_argument("--fold_num", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    # note that we will SHUFFLE by default
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--use_clean_vid", action="store_true", default=False)
    parser.add_argument(
        "--loss_scale",
        nargs='+',
        help="scale loss per class, in the form of --loss_scale a b c d")

    parser.add_argument("--loss_reweighting",
                        action="store_true",
                        default=False)
    # this is the beta parameter in the softmax distribution we will
    # use, controls the sharpness of the reweighting, i.e B -> infinity ->
    # one hot
    parser.add_argument("--reweight_beta", type=float, default=1)

    # model args
    parser.add_argument("--model",
                        type=str,
                        default="R3D18",
                        choices=all_models())
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lr_step", type=int, default=50)
    parser.add_argument("--dropout_prob", type=float, default=0)
    args = parser.parse_args()
    return args


def to_loader(dset,
              num_workers,
              batch_size,
              pin_memory,
              sampler=None,
              shuffle=False):
    return DataLoader(
        dset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=utils.VideoCollateFnSelector("stack_or_combine"),
        shuffle=shuffle,
        pin_memory=pin_memory,
        sampler=sampler)


def get_covid_dset(args,
                   split,
                   labels_to_load=(0, 1, 2),
                   spatial_transforms=None,
                   temporal_transforms=None):

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split of {split}")

    if args.use_clean_vid:
        split = split + "_clean"
        data_dir = os.path.join(args.data_dir, "clean")
    else:
        data_dir = os.path.join(args.data_dir, "original")

    fold_num = args.fold_num
    # -1 for don't use folding, for backwards compatibility only
    if fold_num == -1:
        csv_path = os.path.join(args.split_dir, f"{split}.csv")
    else:
        csv_path = os.path.join(args.split_dir, "folds",
                                f"{split}_fold_{fold_num}.csv")

    if not os.path.exists(csv_path):
        raise ValueError(f"File at {csv_path} doesn't exist")

    if not os.path.exists(data_dir):
        raise ValueError(f"File at {data_dir} doesn't exist")

    return COVIDDataset(csv_path,
                        data_dir,
                        labels_to_load=labels_to_load,
                        spatial_transforms=spatial_transforms,
                        temporal_transforms=temporal_transforms)


def prepare_dsets(args):
    transforms = get_transform(args.transform)
    train_dset = get_covid_dset(
        args,
        "train",
        spatial_transforms=transforms.spatial["train"],
        temporal_transforms=transforms.temporal["train"])

    val_dset = get_covid_dset(args,
                              "val",
                              spatial_transforms=transforms.spatial["val"],
                              temporal_transforms=transforms.temporal["val"])
    return train_dset, val_dset


def prepare_loaders(args, result_dir):
    train_dset, val_dset = prepare_dsets(args)

    train_loader = to_loader(train_dset,
                             args.num_workers,
                             args.batch_size,
                             args.pin_memory,
                             shuffle=args.shuffle)

    val_loader = to_loader(val_dset,
                           pin_memory=True,
                           shuffle=False,
                           batch_size=1,
                           num_workers=1)

    save_one_video(train_loader, os.path.join(result_dir,
                                              "training_sample.mp4"))
    save_one_video(val_loader, os.path.join(result_dir, "val_sample.mp4"))

    yield train_loader, val_loader


def move_and_process_input(batch):
    """Do a transpose to go from NxTxHxWxC -> NxCxHxWxT
    """
    x, y = batch
    x = x.to(device).float()
    y = torch.as_tensor(y).to(device)
    x = x.permute(0, -1, 1, 2, 3)
    return x, y


def train(state, train_loader, val_loader, loss_fn):
    cur_epoch, model, optimizer, scheduler, val_loss_list, ckpt_manager, prev_epoch_loss_list = state

    train_metrics, val_metrics = MetricsCalculator(), MetricsCalculator()

    model.train()

    while cur_epoch < args.num_epochs + 1:
        log_and_print("epoch {}/{}".format(cur_epoch, args.num_epochs))

        if args.loss_reweighting:
            train_dset, _ = prepare_dsets(args)
            num_dset = len(train_dset)
            if cur_epoch == 1:
                prev_epoch_loss_list.extend(torch.zeros(num_dset).tolist())

            weights = torch.softmax(args.reweight_beta *
                                    torch.as_tensor(prev_epoch_loss_list),
                                    dim=0)
            weighted_loss_sampler = sampler.WeightedRandomSampler(
                weights, num_samples=num_dset, replacement=True)

            train_loader = to_loader(train_dset,
                                     args.num_workers,
                                     args.batch_size,
                                     args.pin_memory,
                                     shuffle=False,
                                     sampler=weighted_loss_sampler)
            prev_epoch_loss_list.clear()

        # training
        for i, batch in enumerate(train_loader):
            inputs, labels = move_and_process_input(batch)
            optimizer.zero_grad()
            prediction_tensor = model(inputs)
            loss = loss_fn(prediction_tensor, labels)
            if args.loss_reweighting:
                prev_epoch_loss_list.extend(loss.detach().tolist())
                loss = loss.mean()
            loss.backward()
            train_metrics.add_batch(loss.item(), prediction_tensor.detach(),
                                    labels)
            optimizer.step()

        scheduler.step()

        # compute stats
        train_loss, train_accuracy, train_confusion_matrix, train_preds, train_ground_truths = (
            train_metrics.compute_and_reset(reset=True))

        # validation
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_loader):
                inputs, labels = move_and_process_input(batch)
                prediction_tensor = model(inputs)
                loss = loss_fn(prediction_tensor, labels)
                val_metrics.add_batch(loss.item(), prediction_tensor.detach(),
                                      labels)
            model.train()

        val_loss, val_accuracy, val_confusion_matrix, val_preds, val_ground_truths = (
            val_metrics.compute_and_reset(reset=True))

        print_str = ("Epoch {} with train loss: {} train accuracy: {} "
                     "val loss: {} val accuracy: {}").format(
                         cur_epoch, train_loss, train_accuracy, val_loss,
                         val_accuracy)
        log_and_print(print_str)

        val_loss_list.append(val_loss)
        ckpt_manager.save(global_step=cur_epoch, metric=val_accuracy.item(), do_logging=True)
        label_remap = {0: 0, 1: 1, 2: 0}
        if args.use_wandb:
            wandb.log({
                "epoch":
                    int(cur_epoch),
                "train_loss":
                    train_loss,
                "train_accuracy":
                    train_accuracy,
                "train_confusion_matrix":
                    wandb.plot.confusion_matrix(
                        y_true=train_ground_truths,
                        preds=train_preds,
                        class_names=list(COVIDDataset.key_to_idx.keys())),
                "val_loss":
                    val_loss,
                "val_accuracy":
                    val_accuracy,
                "val_confusion_matrix":
                    wandb.plot.confusion_matrix(
                        y_true=val_ground_truths,
                        preds=val_preds,
                        class_names=list(COVIDDataset.key_to_idx.keys())),
                "precision":
                    compute_precision(val_ground_truths, val_preds),
                "recall":
                    compute_recall(val_ground_truths, val_preds),
                "f1":
                    compute_f1(val_ground_truths, val_preds),
                "precision_covid":
                    compute_precision(val_ground_truths,
                                      val_preds,
                                      label_remap=label_remap),
                "recall_covid":
                    compute_recall(val_ground_truths,
                                   val_preds,
                                   label_remap=label_remap),
                "f1_covid":
                    compute_f1(val_ground_truths,
                               val_preds,
                               label_remap=label_remap),
            })

        cur_epoch.add_(1)

    return val_loss_list


def main(args):

    run_name = args.run_name
    result_dir = os.path.join(args.results_dir, args.model, run_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.model, run_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_class = get_model(args.model)
    model = model_class(args.num_classes,
                        args.pretrained,
                        dropout_prob=args.dropout_prob).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step)

    loss_scale_weight = None
    if args.loss_scale is not None:
        if len(args.loss_scale) != args.num_classes:
            raise ValueError(
                "the number of scale weights \"{}\", provided to"
                " --loss_scale must be equal to the number of classes \"{}\"".
                format(len(args.loss_scale, args.num_classes)))
        args.loss_scale = [float(i) for i in args.loss_scale]
        loss_scale_weight = torch.tensor((args.loss_scale)).to(device)

    reduction = "none" if args.loss_reweighting else "mean"

    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_scale_weight,
                                        reduction=reduction)

    # initialize wandb
    if args.use_wandb:
        wandb_id_path = Path(os.path.join(result_dir, "_wandb_id.txt"))
        init_or_resume_wandb_run(wandb_id_path,
                                 project_name=args.wandb_project_name,
                                 entity=args.wandb_entity,
                                 run_name=run_name,
                                 config=args)
        wandb.watch(model)

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=os.path.join(result_dir, "run_log.txt"),
                        filemode='a')

    loader_iter = prepare_loaders(args, result_dir)

    state = create_or_restore_training_state(model, optimizer, scheduler,
                                             checkpoint_dir)

    for train_loader, val_loader in loader_iter:
        val_loss_list = train(state, train_loader, val_loader, loss_fn)
        print(max(val_loss_list))


if __name__ == "__main__":
    args = prepare_args()
    main(args)
