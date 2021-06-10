import logging
import signal
import time

from pathlib import Path
import torch
import numbers

from .base_interface import SaveableInterface

_sig_hander_map = {}


def check_compliance(objects):
    """Checks if the passed in objects are complient with the saveable
        interface, note that we do NOT use isinstance against
        the SaveableInterface here so that PyTorch models
        and optimizers will work out of the box with our interface
    """
    for name, obj in objects.items():
        if isinstance(obj, SaveableInterface):
            continue
        # explicitly check for required methods
        for attr_to_check in {"state_dict", "load_state_dict"}:
            if not hasattr(obj, attr_to_check):
                raise TypeError("{} of {} needs to implement the {} fn".format(
                    obj, type(obj), attr_to_check))


def disable_signal(*signals):
    global _sig_hander_map
    try:
        for sig in signals:
            orig_handler = signal.getsignal(sig)
            _sig_hander_map[sig] = orig_handler
            # replace with a no-op handler
            signal.signal(signal.SIGINT, lambda _sig, _frame: None)
    except ValueError:
        # Signal throws a ValueError if we're not in the main thread.
        orig_handler = None


def restore_signal(*signals):
    global _sig_hander_map
    for sig, handler in _sig_hander_map.items():
        signal.signal(sig, handler)

    _sig_hander_map.clear()


class SavableCollection:
    """This manages a collection of Savables and you can add your classes
        that either inherit from SaveableInterface or have state_dict
        and load_state_dict implemented to properly save a checkpoint
        in the event of preemption
    """

    def __init__(self, **saveable_classes):
        check_compliance(saveable_classes)
        # we simply use the __dict__attribute since the collection
        # class should only contain a collection of savables.
        self.__dict__.update(saveable_classes)

    def add_managee(self, **saveable_classes):
        """Adds an extra saveable_class to manage
        """
        check_compliance(saveable_classes)
        for name in saveable_classes:
            if name in self.__dict__:
                logging.warning("Attribute of SavableCollection {} already "
                                "exists, will be replaced".format(name))

        self.__dict__.update(saveable_classes)

    def collect_all(self):
        """Goes through the list of objects that implements functions specified
            in the SaveableInterface and get their state dictionary.
        """
        save_dict = {}
        for name, saveable_class in self.__dict__.items():
            save_dict[name] = saveable_class.state_dict()
        return save_dict

    def save(self, save_path, temp_name="temp_ckpt", exist_ok=False):
        """A safer way of saving the state with path replacement
        """
        save_path = Path(save_path).resolve()
        if save_path.is_dir():
            raise ValueError(
                "save_path {} must be a file, not a directory".format(
                    save_path))

        if not exist_ok and save_path.exists():
            raise ValueError(
                "save_path {} already exists but exist_ok is set to False".
                format(save_path))

        # in case the save_path doesn't exist yet
        save_path.parent.mkdir(exist_ok=True)
        temp_path = Path(save_path.stem + "." + temp_name)

        torch.save(self.collect_all(), temp_path)

        # Path().replace() is just os.replace()/rename(), and this call is Posix compliant
        # which means Path().replace() is also atomic (on Posix OS's)
        temp_path.replace(save_path)

        return save_path

    def load(self, load_path, assert_consumed=True):
        """Note that this function assumes that you've instantiated and added
            all the classes you put in self.tracked in the exact same ORDER
            as the saved checkpoint
        """
        store_dict = torch.load(load_path)
        to_be_consumed = set(self.__dict__.keys())
        for name, state in store_dict.items():
            self.__dict__[name].load_state_dict(state)
            to_be_consumed.remove(name)
        if assert_consumed:
            assert (len(to_be_consumed) == 0)


class CheckpointManager(object):
    """Push-button checkpoint manager for most training scenarios
     This class allows for a clean interface to start-or-restore
     a checkpoint and handles deletion/creation of files.

     The checkpoint uses a directory to manage all the checkpoint
     files and history. It also keeps track of the # of checkpoint
     files and the latest one based the internal tracking.

  """

    # internal struct to organize thing better
    class Entry(object):

        def __init__(self, step, checkpoint, metric):
            self.step = step
            self.checkpoint = checkpoint
            self.metric = metric

    ckpt_suffix = "pt"
    best_ckpt_suffix = "best_ckpt"
    temp_ckpt_suffix = "temp_ckpt"

    def __init__(
        self,
        checkpoint,
        directory,
        max_to_keep,
        keep_best=None,
        checkpoint_prefix="checkpoint",
        checkpoint_interval=None,
        init_fn=None,
    ):
        """
    Args:
        checkpoint (SavableCollection): an instance of SavableCollection
            to manage (store state_dict to and restore state_dict from)

        directory (str): The path to a directory in which to write checkpoints. The
            checkpoint manager uses checkpoint suffixes to maintain additional
            information about the checkpoints (global step ID)

        max_to_keep (int): the number of checkpoints to keep. The number is
            maintained in FIFO order.

        keep_best (str [max|min], optional): If a string of either "max" or "min"
            is provided, then the manager takes a "metric" and keeps the best
            performing checkpoint during "save" accordingly. This best
            checkpoint is not subject to the max_to_keep value.

        checkpoint_interval (int, optional): Minimum time between two checkpoints, in
            seconds. (note that if "save" is called within checkpoint_interval
            it becomes an no-op)

        init_fn (callable, optional): This init function will be called when there does
            not exist a checkpoint in load_latest_checkpoint

    Raises:
      ValueError: If `max_to_keep` or checkpoint_interval (when specified)
        is not a positive integer.
    """

        # attributes for managing the checkpoint
        self.checkpoint = checkpoint
        self.directory = Path(directory)
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_interval = checkpoint_interval

        # attributes of the manager and initial states
        self.checkpoints_list = []
        self.best_checkpoint = None
        self.lastest_checkpoint_time = -1
        self.init_fn = init_fn if init_fn is not None else lambda: None

        # sanity checks
        if keep_best is not None and keep_best not in ("max", "min"):
            raise ValueError("\"keep_best\" provided must be either \"max\" "
                             "or \"min\", not \"{}\"".format(keep_best))
        if max_to_keep <= 0:
            raise ValueError(
                "max_to_keep must be greater than 0, got {}".format(
                    max_to_keep))

        if checkpoint_interval is not None and checkpoint_interval <= 0:
            raise ValueError(
                "checkpoint_interval must be greater than 0, got {}".format(
                    checkpoint_interval))

        if "." in checkpoint_prefix:
            raise ValueError("checkpoint_prefix cannot contain \".\"")

        if self.directory.exists():
            if self.directory.is_dir():
                # implicitly populates:
                # self.checkpoints_list
                # self.lastest_checkpoint_time (expressed in ctime, seconds)
                self.initialize_state()
            else:
                raise ValueError(
                    "Provided directory {} must be a directory, not a file".
                    format(self.directory))

        else:
            self.directory.mkdir(parents=True)

        # if there are no checkpoints, then this must be a new run
        # so we need to call the init_fn which can
        # include something like SaveableWandBRun.initialize_run which requires
        # an initial wandb.init call internally if previous checkpoints
        # don't exist
        if len(self.checkpoints_list) == 0:
            self.init_fn()

    def initialize_state(self):

        checkpoints = []
        dir_empty = True
        latest_time = -1

        for file in self.directory.iterdir():
            if not file.is_file():
                raise ValueError(
                    "File {} under {} is not a regular file".format(
                        file, self.directory))

            dir_empty = False

            if file.name.startswith(self.checkpoint_prefix):
                # this assumed a unified save standard
                prefix, global_step, metric, suffix = file.name.split(".")
                if suffix == CheckpointManager.temp_ckpt_suffix:
                    logging.info("Removed temporary checkpoint {}" \
                        " This is possibly due to preemption happening during saving checkpoint"
                        .format(file))
                    file.unlink()
                    continue

                global_step = int(global_step)
                metric = float(metric.replace(
                    "__", ".")) if metric != "none" else None

                if suffix == CheckpointManager.best_ckpt_suffix:
                    self.best_checkpoint = CheckpointManager.Entry(
                        global_step, file, metric)
                    continue

                last_modified = file.stat().st_mtime
                if last_modified > latest_time:
                    latest_time = last_modified

                checkpoints.append(
                    CheckpointManager.Entry(global_step, file, metric))

        if not dir_empty and latest_time == -1:
            logging.warning(
                "directory {} not empty but no checkpoint file found with name {}"
                .format(self.directory, self.checkpoint_prefix))
            return

        # sort the list by global step
        self.checkpoints_list = list(sorted(checkpoints, key=lambda x: x.step))
        self.lastest_checkpoint_time = latest_time

    @property
    def global_step(self):
        return 0 if len(self.checkpoints_list) == 0 else \
            self.checkpoints_list[-1].step

    @property
    def latest_checkpoint(self):
        return None if len(self.checkpoints_list) == 0 else \
            self.checkpoints_list[-1].checkpoint

    def load_latest_checkpoint(self):
        if self.latest_checkpoint is None:
            logging.info("No checkpoint loaded")
        else:
            self.checkpoint.load(self.latest_checkpoint)
            logging.info("loaded latest checkpoint from {}".format(
                self.latest_checkpoint))

    def save(self, global_step=None, metric=None, do_logging=False):

        if metric is not None and not isinstance(metric, numbers.Number):
            raise TypeError("metric must be a number, not {}".format(
                type(metric)))

        if self.keep_best and metric is None:
            raise ValueError("metric must be specified when keep_best=True")

        if isinstance(global_step, SaveableInterface):
            global_step = global_step.get_native_value()

        metric_str = "none" if metric is None else "{:.2f}".format(
            metric).replace(".", "__")

        if self.checkpoint_interval is not None:
            elapsed = time.time() - self.lastest_checkpoint_time
            if elapsed < self.checkpoint_interval:
                logging.debug(
                    "checkpoint skipped due to checkpoint_interval={}s, "
                    "but only {}s has elapsed since last checkpoint".format(
                        elapsed, self.checkpoint_interval))
                return

        if global_step is not None:
            if global_step <= self.global_step:
                raise ValueError(
                    "global_step {} is less or equal than manager's current global step {}"
                    .format(global_step, self.global_step))
            else:
                save_step = global_step
        else:
            save_step = self.global_step + 1

        ckpt_name = "{}.{}.{}.{}".format(self.checkpoint_prefix, save_step,
                                         metric_str,
                                         CheckpointManager.ckpt_suffix)
        save_path = self.directory / ckpt_name

        # Best effort: these updates should not be interrupted
        disable_signal(signal.SIGINT, signal.SIGTERM)

        saved = self.checkpoint.save(save_path, exist_ok=False,
                        temp_name=CheckpointManager.temp_ckpt_suffix)
        if do_logging:
            logging.info("Checkpoint saved under {}".format(saved))

        # update states
        self.checkpoints_list.append(
            CheckpointManager.Entry(save_step, saved, metric))
        self.lastest_checkpoint_time = int(time.time())

        # remove extra checkpoints
        if len(self.checkpoints_list) > self.max_to_keep:
            self.checkpoints_list[0].checkpoint.unlink()
            self.checkpoints_list.pop(0)

        # update the best checkpoint
        if self.keep_best is not None:
            save = False
            if self.best_checkpoint is None:
                save = True
            elif self.keep_best == "max" and metric > self.best_checkpoint.metric:
                save = True
            elif self.keep_best == "min" and metric < self.best_checkpoint.metric:
                save = True

            if save:
                best_ckpt_name = "{}.{}.{}.{}".format(
                    self.checkpoint_prefix, save_step, metric_str,
                    CheckpointManager.best_ckpt_suffix)
                save_path = self.directory / best_ckpt_name

                best_saved = self.checkpoint.save(save_path, exist_ok=False,
                        temp_name=CheckpointManager.temp_ckpt_suffix)

                if self.best_checkpoint is not None:
                    self.best_checkpoint.checkpoint.unlink()
                self.best_checkpoint = CheckpointManager.Entry(
                    save_step, best_saved, metric)

        restore_signal()
