import os
from typing import Optional, List, Tuple, Dict, Any
from src.models import NCA, DINCA
from src.loss import WeightedPixelLoss, FilterAwareWeightedPixelLoss
from src.loader import DataLoader, ModelSetupManager
from src.analysis_manager import AnalysisManager
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random
import numpy as np
import imageio
import pickle
from tqdm import tqdm
import wandb



class TrainingManager:
    def __init__(self, config: Dict[str, Any], data_loader: DataLoader, model_loader: ModelSetupManager):
        """
        Initialize the TrainingManager with the given configuration.

        :param config: Configuration dictionary for training.
        :param data_loader: DataLoader instance for loading data.
        :param model_loader: ModelSetupManager instance for setting up models.
        """
        self.config = config
        self.output_folder_path = config['output_folder']
        self.model_path = config['model_path']
        self.loss_path = config['loss_path']
        self.device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
        self.channel_mode = config['channel_mode']

        self.n_epochs = config['training']['epochs']
        self.simulation_steps = config['training']['simulation_steps']
        self.save_interval = config['training']['save_interval']
        self.batch_size = config['training']['batch_size']
        self.enable_pruning = config['training']['pruning']['enable']
        if self.enable_pruning:
            self.pruning_percentiles = {sched['iteration']: sched['percentile'] for sched in config['training']['pruning']['schedule']}
        
        self.loss_history = []

        self.data_loader = data_loader
        self.model_loader = model_loader

        self.wandb_disabled = config['wandb_disabled']

        if not self.wandb_disabled:
            wandb.init(project=config['wandb_project_name'], config=config, name=config['wandb_experiment_name'])

        self._setup_objects()


    def train(self):
        best_loss = float('inf')
        x_init_batch = self.x_init.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        np.save(os.path.join(self.output_folder_path, 'init_state.npy'), x_init_batch.detach().cpu().numpy())

        for i in tqdm(range(len(self.loss_history) + 1, self.n_epochs + 1)):
            x_outs = [x_init_batch]
            for step_range in self.simulation_steps:
                n = random.randint(step_range[0], step_range[1])
                x_out = self.model(x_outs[-1], n)
                x_outs.append(x_out)

            total_loss, loss_elements = self.loss_evaluator.compute(self.model, x_outs[1:], self.x_refs)
            self.loss_history.append(loss_elements)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.model.freeze_pruned_weights()
            
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.model.state_dict(), os.path.join(self.output_folder_path, 'best_model.pth'))

            if self.enable_pruning and i in self.pruning_percentiles:
                self.model.prune_weights(self.pruning_percentiles[i])

            if not self.wandb_disabled:
                wandb.log({"total_loss": total_loss, **loss_elements})

            if i % self.save_interval == 0:
                torch.save(self.model.state_dict(), self.model_path)
                with open(self.loss_path, 'wb') as file:
                    pickle.dump(self.loss_history, file)

                analysis_manager = AnalysisManager(self.output_folder_path)
                analysis_manager.get_loss_plot(self.loss_history)
                steps_sum = sum([step_range[1] for step_range in self.simulation_steps])
                analysis_manager.get_simulation_plot(self.model, x_init_batch, steps_sum)
                analysis_manager.get_batch_plot(x_outs, self.x_init, self.x_refs, self.channel_mode)
                analysis_manager.get_filter_values_plot(self.model.get_filters())


    def _setup_objects(self):
        """
        Setup the training environment including data, model, optimizer, scheduler, and loss evaluator.
        """
        self._setup_data()
        self._setup_model()
        self._setup_optimizer_and_scheduler()
        self._setup_loss_evaluator()


    def _setup_data(self):
        config = self.config
        self.normalized_images = self.data_loader.load_images(config['image_folder'])
        self.x_init = self.data_loader.setup_init_state(
            n_channels=config['automaton_settings']['channels']['number'],
            alteration_method=config['initial_state']['alteration_method'],
            padding_size=config['automaton_settings']['channels']['padding']['size'],
            padding_values=config['automaton_settings']['channels']['padding']['values'],
            use_predefined_shape=config['initial_state']['use_predefined_shape'],
            shape_type=config['initial_state']['shape_type']
        )
        self.x_init = torch.tensor(self.x_init).to(self.device).float()
        self.x_refs = torch.tensor(self.data_loader.get_reference_states()).to(self.device)


    def _setup_model(self):
        config = self.config
        model_type = config['model']['type']
        common_model_args = {
            'n_channels': config['automaton_settings']['channels']['number'],
            'fire_rate': config['automaton_settings']['cell_fire_rate'],
            'init_weight_factor': config['automaton_settings']['neural_net']['initial_weight_factor'],
            'value_range': config['automaton_settings']['channels']['value_range'],
            'filter_size': config['filters']['size'],
            'n_filters': config['filters']['number'],
            'learnable_filters': config['filters']['learnable'],
            'padding_mode': config['automaton_settings']['channels']['padding']['mode'],
            'filters_custom_init': config['filters']['custom_init'],
            'device': self.device,
            'filters_init_values': config['filters']['initial_values'],
            'padding_values': config['automaton_settings']['channels']['padding']['values']
        }
        if model_type == 'NCA':
            common_model_args['hidden_size'] = config['nca_specific']['hidden_layer_size']
            self.model = self.model_loader.setup_NCA(**common_model_args)
        elif model_type == 'DINCA':
            common_model_args['term_max_power'] = config['dinca_specific']['term_max_power']
            self.model = self.model_loader.setup_DINCA(**common_model_args)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if config['training']['load_weights']:
            self.model = self.model_loader.load_weights(config['model_path'])
            with open(self.loss_path, 'rb') as file:
                self.loss_history = pickle.load(file)

        self.model.to(self.device)


    def _setup_optimizer_and_scheduler(self):
        config = self.config
        self.optimizer = self.model_loader.setup_optimizer(
            opt_type=config['automaton_settings']['neural_net']['optimizer']['type'],
            opt_lr=config['automaton_settings']['neural_net']['optimizer']['learning_rate'],
            opt_betas=config['automaton_settings']['neural_net']['optimizer']['betas']
        )
        self.scheduler = self.model_loader.setup_scheduler(
            sch_type=config['automaton_settings']['neural_net']['scheduler']['type'],
            sch_gamma=config['automaton_settings']['neural_net']['scheduler']['decay_rate']
        )


    def _setup_loss_evaluator(self):
        config = self.config
        if config['filters']['use_moment_constraints']:
            prescribed_moments = torch.tensor(config['filters']['prescribed_moments'], dtype=torch.float32).to(self.device)
            self.loss_evaluator = self.model_loader.setup_filter_aware_weighted_pixel_loss(
                torch_loss_name=config['loss']['type'],
                padding_size=config['automaton_settings']['channels']['padding']['size'],
                channel_mode=config['channel_mode'],
                reg_factor=config['loss']['regularization_factor'],
                prescribed_moments=prescribed_moments,
                filters_factor=config['loss']['filters_factor']
            )
        else:
            self.loss_evaluator = self.model_loader.setup_weighted_pixel_loss(
                torch_loss_name=config['loss']['type'],
                padding_size=config['automaton_settings']['channels']['padding']['size'],
                channel_mode=config['channel_mode'],
                reg_factor=config['loss']['regularization_factor']
            )
