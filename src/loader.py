import os
import numpy as np
import imageio
from typing import List, Tuple
from src.models import NCA, DINCA
from src.loss import WeightedPixelLoss, FilterAwareWeightedPixelLoss
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler



class DataLoader:
    def __init__(self):
        pass


    def get_reference_states(self):
        assert (self.reference_states is not None), "The initial state must be set up first."
        return self.reference_states


    def load_images(self, folder_path: str):
        self.raw_images = self._load_raw_images(folder_path)
        self.raw_images = self.raw_images[:, :, :, :3] # discard alpha channel
        self.normalized_images = self.raw_images / 255.0
        return self.normalized_images


    def setup_init_state(self, n_channels: int, alteration_method: List[str], padding_size: int, padding_values: List[int],
                         use_predefined_shape: bool, shape_type: str = None):
        assert(self.normalized_images is not None), "The images must be loaded first."
        width, height = self.normalized_images.shape[1], self.normalized_images.shape[2]

        if use_predefined_shape:
            init_state_raw = self._setup_init_state_from_predefined_shape(width, height, n_channels, shape_type)
            self.reference_states = self.normalized_images
        else:
            init_state_raw = self._setup_init_state_from_first_image(width, height, n_channels)
            self.reference_states = self.normalized_images[1:, :, :, :] # first image not referential as it's used as init state

        # init_state_raw is of shape [width, height, n_channels]
        n_channels = init_state_raw.shape[2]
        self.initial_state = np.array([self._state_alternation(init_state_raw[:, :, c], alteration_method[c])
                                       for c in range(n_channels)])
        p = padding_size
        self.initial_state = [np.pad(x_c, ((p, p), (p, p)), mode='constant', constant_values=val_c)
                              for x_c, val_c in zip(self.initial_state, padding_values)]
        self.initial_state = np.array(self.initial_state).transpose(1, 2, 0)
        return self.initial_state


    def _load_raw_images(self, path: str):
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        files = sorted(files)
        imgs = np.array([np.array(imageio.imread(path).astype(np.float32)) for path in files])
        return imgs


    def _state_alternation(self, state: np.ndarray, mode: str):
        if mode == 'zeros':
            return np.zeros_like(state)
        elif mode == 'original':
            return state
        elif mode == 'flipped':
            return 1.0 - state

    def _get_shape(self, width, height, shape_type):
        def gen_noise():
            return np.random.rand(width, height)

        def gen_gradient():
            step_size = 1 / (width - 1)
            arr = np.zeros((width, height))
            for i in range(width):
                arr[i, :] = i * step_size
            return arr

        def gen_ball_gradient():
            step_size = 1 / (width / 2)
            arr = np.zeros((width, height))
            x_middle, y_middle = int(width / 2), int(height / 2)
            for i in range(width):
                for j in range(height):
                    distance = max(abs(x_middle - i), abs(y_middle - j))
                    arr[i, j] = 1 - (step_size * distance)
            return arr

        def gen_stripes():
            arr = np.zeros((width, height))
            for i in range(1, width + 1, 1):
                if i % 2 == 0:
                    arr[i - 1:i, :] = 1
                else:
                    arr[i - 1:i, :] = 0
            return arr

        state_gen_fn = {
            'noise': gen_noise,
            'gradient': gen_gradient,
            'ball_gradient': gen_ball_gradient,
            'stripes': gen_stripes
        }
        return state_gen_fn[shape_type]()


    def _setup_init_state_from_predefined_shape(self, width: int, height: int, n_channels: int, shape_type: str):
        init_state_raw = self._get_shape(width, height, shape_type)
        init_state_raw = np.repeat(init_state_raw[:, :, np.newaxis], n_channels, axis=-1)
        return init_state_raw


    def _setup_init_state_from_first_image(self, width: int, height: int, n_channels: int):
        init_state_raw = self.normalized_images[0, :, :, :]
        if init_state_raw.shape[2] > n_channels:
            init_state_raw = np.mean(init_state_raw, axis=2)
            init_state_raw = np.repeat(init_state_raw[:, :, np.newaxis], n_channels, axis=-1).astype(np.float32)
        elif n_channels > init_state_raw.shape[2]:
            channels_to_add = n_channels - init_state_raw.shape[2]
            init_state_raw = np.concatenate((init_state_raw, np.zeros((width, height, channels_to_add))), axis=2).astype(np.float32)
        return init_state_raw



class ModelSetupManager:
    def __init__(self):
        pass


    def setup_NCA(self, hidden_size: int, n_channels: int, fire_rate: float, init_weight_factor: float,
                  value_range: Tuple[float, float], filter_size: int, n_filters: int, learnable_filters: bool,
                  padding_mode: str, filters_custom_init: bool, device: torch.device,
                  filters_init_values: List[List[List[float]]] = None,
                  padding_values: float = None):
        self.model = NCA(hidden_size, n_channels, fire_rate, init_weight_factor, value_range, filter_size,
                         n_filters, learnable_filters, padding_mode, filters_custom_init, device,
                         filters_init_values, padding_values)
        return self.model


    def setup_DINCA(self, term_max_power: int, n_channels: int, fire_rate: float, init_weight_factor: float,
                    value_range: Tuple[float, float], filter_size: int, n_filters: int, learnable_filters: bool,
                    padding_mode: str, filters_custom_init: bool, device: torch.device,
                    filters_init_values: List[List[List[float]]] = None,
                    padding_values: float = None):
        self.model = DINCA(term_max_power, n_channels, fire_rate, init_weight_factor, value_range, filter_size,
                         n_filters, learnable_filters, padding_mode, filters_custom_init, device,
                           filters_init_values, padding_values)
        return self.model


    def load_weights(self, path):
        assert(self.model is not None), "The model must be set up first."
        self.model.load_state_dict(torch.load(path))
        return self.model


    def setup_optimizer(self, opt_type: str, opt_lr: float, opt_betas: Tuple[float, float]):
        assert(self.model is not None), "The model must be set up first."
        self.optimizer = getattr(optim, opt_type)(
            self.model.parameters(),
            lr=opt_lr,
            betas=opt_betas
        )
        return self.optimizer


    def setup_scheduler(self, sch_type: str, sch_gamma: float):
        assert(self.optimizer is not None), "The optimizer must be set up first."
        self.scheduler = getattr(lr_scheduler, sch_type)(
            self.optimizer,
            gamma=sch_gamma
        )
        return self.scheduler


    def setup_weighted_pixel_loss(self, torch_loss_name: str, padding_size: int, channel_mode: str, reg_factor: float):
        loss_fn = getattr(F, torch_loss_name, None)
        self.loss_evaluator = WeightedPixelLoss(reg_factor, loss_fn, padding_size, channel_mode)
        return self.loss_evaluator


    def setup_filter_aware_weighted_pixel_loss(self, torch_loss_name: str, padding_size: int, channel_mode: str, reg_factor: float,
                                               prescribed_moments: torch.Tensor, filters_factor: float):
        loss_fn = getattr(F, torch_loss_name, None)
        self.loss_evaluator = FilterAwareWeightedPixelLoss(reg_factor, prescribed_moments, filters_factor,
                                                           loss_fn, padding_size, channel_mode)
        return self.loss_evaluator
