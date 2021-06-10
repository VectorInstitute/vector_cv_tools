import math
import random
import numbers
import warnings
from typing import Optional
from functools import total_ordering
from collections.abc import MutableSequence
import wandb
import torch
import numpy as np
from .base_interface import SaveableInterface, validate_dict
from .saveable_sampler import SaveableSampler
from ..utils.misc import set_all_randomness


class SaveableWandBRun(SaveableInterface):
    """This class wraps around the wandb run to allow for
        the resuming of your runs if you decide to use
        wandb

    Note that because we can only call wandb.init once,
        this class will requires you to do the following:
            SaveableCollection(..., ..., mywandb_run=SaveableWandBRun)
            ...
            manager.load_latest_checkpoint()
            SaveableWandBRun.initialize_run()

        i.e we need to explicitly call .initialize_run() AFTER
            we call the load on checkpoint manager
    """

    def __init__(self, **wandb_kwargs):
        """You should pass in everything you would otherwise
            pass into the wandb init function via this wandb_kwargs,
            besides from the resume_id
        """
        self.key = "run_id"

        if "resume" in wandb_kwargs:
            del wandb_kwargs["resume"]
            warnings.warn(
                "The key resume is detected in the wandb_kwargs. Removing "
                "this key to allow for proper resuming")

        self.wandb_kwargs = wandb_kwargs
        self.resume_id = None
        self.wandb_config = None

    def initialize_run(self):
        """This function actually initializes the wandb run,
            note that we don't do this init because we can
            only call wandb.init once
        """
        run = wandb.init(**self.wandb_kwargs, resume=self.resume_id)
        self.resume_id = run.id
        self.wandb_config = run.config
        return run

    def get_wandb_config(self):
        return self.wandb_config

    def state_dict(self):
        dict_to_save = {self.key: self.resume_id}
        return dict_to_save

    def load_state_dict(self, state_dict):
        validate_dict(state_dict, self.key)
        self.resume_id = state_dict[self.key]
        self.initialize_run()

    def __repr__(self):
        return "{} with resume id {}".format(self.__class__.__name__,
                                             str(self.resume_id))


class SaveableRNG(SaveableInterface):
    """A class that handles saving of the RNG state with torch, numpy
        and random library

    VERY IMPORTANT NOTE:
        *** This does NOT keep track of the randomness in torch.cuda ***
        *** it also doesn't set the randomness for torch.cuda ***
    """

    def __init__(self, seed, set_seed_on_init=True, set_seed_on_load=True):
        if set_seed_on_init:
            set_all_randomness(seed, set_for_cuda=False)
        self.seed = seed
        self.keys = {
            "seed",
            "np_rng",
            "py_rng",
            "torch_rng",
        }
        self.set_seed_on_load = set_seed_on_load

    def state_dict(self):
        """Returns the state state_dict with the current RNG for PyTorch
            Numpy and Random libaries

        This does not consider CUDA random state
        """
        dict_to_save = {
            "seed": self.seed,
            "np_rng": np.random.get_state(),
            "py_rng": random.getstate(),
            "torch_rng": torch.random.get_rng_state(),
        }
        return dict_to_save

    def load_state_dict(self, state_dict):
        """Note that this function will call set_all_randomness again
            to make sure that the randomness is set to the proper seed
            before loading the RNG
        """
        validate_dict(state_dict, *self.keys)
        # restore all the fields
        seed = state_dict["seed"]
        np_rng = state_dict["np_rng"]
        py_rng = state_dict["py_rng"]
        torch_rng = state_dict["torch_rng"]

        if self.set_seed_on_load:
            set_all_randomness(seed, set_for_cuda=False)
        torch.random.set_rng_state(torch_rng)
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        self.seed = seed

    def __repr__(self):
        return "{} wrapper for RNG state with seed {}".format(
            self.__class__.__name__, self.seed)


class SaveableList(MutableSequence, SaveableInterface):
    """Note that everything you add to this class must be
        pickleable!

    This essentially wraps around a list, and you can use it
        as one. Good for tracking losses etc..
    """

    def __init__(self, items: Optional[list] = None):
        self._internal_list = [] if items is None else items
        self.key = "values"

    def state_dict(self):
        dict_to_save = {self.key: self._internal_list}
        return dict_to_save

    def load_state_dict(self, state_dict):
        validate_dict(state_dict, self.key)
        self._internal_list = state_dict[self.key]

    def get_native_value(self):
        """Be careful of the mutating the internal list here, the current class
            instance will still see the mutating change

        If you want a copy of the list, then use deepcopy
        """
        return self._internal_list

    def insert(self, index, value):
        self._internal_list.insert(index, value)

    def __setitem__(self, index, value):
        self._internal_list.__setitem__(index, value)

    def __delitem__(self, index):
        self._internal_list.__delitem__(index)

    def __getitem__(self, index):
        return self._internal_list.__getitem__(index)

    def __len__(self):
        return self._internal_list.__len__()

    def __str__(self):
        return str(self._internal_list)

    def __repr__(self):
        return "{} wrapper for list with value {}".format(
            self.__class__.__name__, str(self._internal_list))


class SaveableDataLoader(torch.utils.data.DataLoader, SaveableInterface):
    """This class wraps around the default PyTorch dataloader, so it's
        possible to save the dataloader between epochs. Note that
        the DataLoader really has no states, so it just calls
        the state_dict and load_state_dict function of the
        SaveableSampler
    """

    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 batch_size=1,
                 shuffle=False,
                 to_save_dataloader_iter=True,
                 **dataloader_kwargs):
        """
        Arguments:
            dataset (torch.utils.data.Dataset): dataset that implements the
                PyTorch dataset interface.
            to_save_dataloader_iter (bool): When true, it will use the latest
                iterator generated by this dataloader to save the state.
                This is useful for the num_workers > 0 schema where the workers
                might prefetch, so having this enabled allows us to "undo" the
                prefetching and actually record the index the iterator
                stopped at.
            dataloader_kwargs: kwargs that are specific to the dataloader like
                num_workers etc..
        """
        # use the saveable sampler
        if "sampler" in dataloader_kwargs:
            if isinstance(dataloader_kwargs['sampler'],
                          torch.utils.data.distributed.DistributedSampler):
                raise NotImplementedError("SaveableDataLoader Does not "
                                          "support Distributed Sampler")
            warnings.warn("The internal implementation of this class uses a "
                          "SaveableSampler. Overwriting the sampler passed in")

        # tell the sampler to shuffle or not here
        saveable_sampler = SaveableSampler(dataset, shuffle=shuffle)
        dataloader_kwargs["sampler"] = saveable_sampler
        dataloader_kwargs["batch_size"] = batch_size

        # we will always give shuffle=False to the dataloader class
        # because this shuffle is encapsulated in the sampler
        # already, i.e shuffle=True/False is dependent
        # purely on the sampler we use to save
        dataloader_kwargs["shuffle"] = False
        super().__init__(dataset, **dataloader_kwargs)

        self._to_save_dataloader_iter = to_save_dataloader_iter
        self._saveable_sampler = saveable_sampler
        self._most_recent_iterator = None

    def __iter__(self):
        iterator = super().__iter__()
        if self._to_save_dataloader_iter:
            self._most_recent_iterator = iterator
        return iterator

    def state_dict(self):
        return self._saveable_sampler.state_dict(self._most_recent_iterator)

    def load_state_dict(self, state_dict):
        self._saveable_sampler.load_state_dict(state_dict)

    def __repr__(self):
        return "{} wrapper around native PyTorch Dataloader".format(
            self.__class__.__name__)


@total_ordering
class SavebleNumber(numbers.Number, SaveableInterface):
    """This class implements most things you would expect from a number
        so it's OK to use it as one. However, since numbers are generally
        immutable, to change the number that really gets saved
        please use [op]_ methods like add_, sub_ etc to mutate the value.

        NOTE:
            The [op]_ like methods will first do the operation
            on the primitive number(i.e self._value) and THEN
            cast it back to the appropriate number type
    """

    def __init__(self,
                 value: numbers.Number,
                 number_type: Optional[type] = None):
        """
        Arguments:
            value (Number): any number type
            number_type (type, Optional): If this argument is None, then
                the class will infer the type of the number passed in by value
                and then cast result of mutating operations to this type.
                If it's not None then the class will use the type passed in
                for casting.
        """
        self.number_type = type(value) if number_type is None else number_type
        self._value = self.number_type(value)
        self.key = "num"

    def state_dict(self):
        dict_to_save = {self.key: self._value}
        return dict_to_save

    def load_state_dict(self, state_dict):
        validate_dict(state_dict, self.key)
        self._value = self.number_type(state_dict["num"])

    def get_native_value(self):
        """Since numbers are immutable, changing the return of this value
            will NOT affect the value in this class
        """
        return self._value

    def set_val(self, value):
        """Overwrites and sets the value of the saveable number. Will
            cast the passed in value to the initialized type
        """
        self._value = self.number_type(value)

    def add_(self, other):
        self._value = self.number_type(self._value + other)

    def sub_(self, other):
        self._value = self.number_type(self._value - other)

    def mul_(self, other):
        self._value = self.number_type(self._value * other)

    def div_(self, other):
        self._value = self.number_type(self._value / other)

    def pow_(self, other):
        self._value = self.number_type(self._value**other)

    def __mul__(self, other):
        return self._value.__mul__(other)

    def __add__(self, other):
        return self._value.__add__(other)

    def __sub__(self, other):
        return self._value.__sub__(other)

    def __mod__(self, other):
        return self._value.__mod__(other)

    def __truediv__(self, other):
        return self._value.__truediv__(other)

    def __floordiv__(self, other):
        return self._value.__floordiv__(other)

    def __abs__(self):
        return self._value.__abs__()

    def __ceil__(self):
        return math.ceil(self._value)

    def __float__(self):
        return self._value.__float__()

    def __floor__(self):
        return math.floor(self._value)

    def __neg__(self):
        return self._value.__neg__()

    def __pos__(self):
        return self._value.__pos__()

    def __pow__(self, other):
        return self._value.__pow__(other)

    def __round__(self, ndigits=0):
        return self._value.__round__(ndigits)

    def __trunc__(self):
        return self._value.__trunc__()

    def __le__(self, other):
        tocmp = (other.get_native_value()
                 if isinstance(other, SavebleNumber) else other)
        return self._value.__le__(tocmp)

    def __eq__(self, other):
        tocmp = (other.get_native_value()
                 if isinstance(other, SavebleNumber) else other)
        return self._value.__eq__(tocmp)

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return ("{} wrapper for {} with value {}".format(
            self.__class__.__name__, self.number_type, str(self._value)))
