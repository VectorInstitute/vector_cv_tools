# system imports
import os
import logging
import glob
from pathlib import Path
import re

# external dependencies
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torchvision.utils import save_image

# imports from vector_cv_tools
from vector_cv_tools import utils as vutils
from vector_cv_tools import transforms as VT
from vector_cv_tools import datasets as vdatasets

################# CHECKPOINTING ######################
from vector_cv_tools.experimental import checkpointing as vckpts
######################################################

# relative imports
from model import ConvVAE, loss_fn

MVTEC_ROOT_DIR = "./datasets/MVTec_AD"
CHECKPOINT_DIR = "./checkpoints"
LOG_FILE = "run_log.out"
TEMP_NAME = "_temp.pt"
lr = 3e-5
num_epochs = 500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_and_print(print_str):
    logging.info(print_str)
    print(print_str)


def to_loader(dset, batch_size=128, num_workers=4):
    # note that this collate fn is needed for all our image datasets
    # as the PyTorch default WILL load the data in the wrong ordering
    return DataLoader(dset,
                      collate_fn=vutils.collate_dictionary_fn,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=True,
                      shuffle=True)


################# CHECKPOINTING ######################


def create_or_restore_training_state():
    """Since this code does not set random seed, saving RNG state is omitted,
        you should definately save RNG state and set RNG seed for your models
    """
    model = ConvVAE().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    cur_epoch = vckpts.SavebleNumber(0)
    # default initialization into []
    losses = vckpts.SaveableList()
    # this queries the rng state from the global namespace
    rng = vckpts.SaveableRNG(666)

    checkpoint = vckpts.SavableCollection(
        model=model,
        optimizer=optimizer,
        cur_epoch=cur_epoch,
        losses=losses,
        rng=rng,
    )

    manager = vckpts.CheckpointManager(
        checkpoint=checkpoint,
        directory="./vae_train_checkpoints",
        max_to_keep=3,
        checkpoint_interval=1200,  # 1200 seconds = 20 mins
    )

    # this call does nothing if there is no checkpoint loaded
    # otherwise it loads from the latest checkpoint in the directory
    # specified above
    manager.load_latest_checkpoint()

    if manager.latest_checkpoint is not None:
        loss_str = "not logged yet" if len(
            checkpoint.losses) == 0 else checkpoint.losses[-1]
        print_str = (
            f"Checkpoint that has finished epoch {checkpoint.cur_epoch} with "
            f"loss {loss_str} is loaded from {manager.latest_checkpoint}")

    else:
        print_str = f"No checkpoints found under {manager.directory}, starting from scratch"

    log_and_print(print_str)
    cur_epoch.add_(1)
    return cur_epoch, model, optimizer, losses, manager


######################################################


def main():
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=LOG_FILE,
                        filemode='a')

    training_transforms = VT.ComposeMVTecTransform(
        [A.Resize(128, 128),
         A.ToFloat(max_value=255),
         ToTensorV2()])
    mvtec_train_dset = vdatasets.MVTec(MVTEC_ROOT_DIR,
                                       split="train",
                                       transforms=training_transforms)

    train_loader = to_loader(mvtec_train_dset)

    ################## CHECKPOINTING ######################
    cur_epoch, model, optimizer, losses, ckpt_manager = \
                     create_or_restore_training_state()
    ######################################################

    model.train()
    model = torch.nn.DataParallel(model)

    while cur_epoch < num_epochs + 1:
        log_and_print("epoch {}/{}".format(cur_epoch, num_epochs))
        train_losses = []
        for i, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = loss_fn(data, recon, mu, logvar)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if i % 100 == 0:
                save_image(recon.detach().cpu(), f"{cur_epoch}_{i}.png")

        mean_loss = sum(train_losses) / len(train_losses) if (
            len(train_losses) > 0) else 0

        print_str = f"Train loss at epoch {cur_epoch} is {mean_loss}"
        log_and_print(print_str)

        ################## CHECKPOINTING ######################
        losses.append(mean_loss)
        # This call collects everything that is collected under the
        # "checkpoint" that was initially passed into the checkpointing
        # manager and handles deletion automatically.
        # Note that this call does nothing if two "saves" are called
        # within checkpoint_interval" seconds.

        # This makes things cleaner. User do not need to keep track of time
        ckpt_manager.save(global_step=cur_epoch, do_logging=True)
        cur_epoch.add_(1)


######################################################

if __name__ == "__main__":
    main()
