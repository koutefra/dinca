import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import itertools
import math



class BaseNCA(nn.Module):
    """
    Base Neural Cellular Automata class. It defines the fundamental structure and behavior of a neural cellular automata.
    This class should be subclassed to implement specific NCA behaviors.
    """
    def __init__(self, n_channels: int, fire_rate: float, init_weight_factor: float, value_range: Tuple[float, float],
                 filter_size: int, n_filters: int, learnable_filters: bool, padding_mode: str, filters_custom_init: bool,
                 device: torch.device, filters_init_values: List[List[List[float]]] = None,
                 padding_values: List[float] = None):
        super().__init__()
        self.n_channels = n_channels
        self.fire_rate = fire_rate
        self.init_weight_factor = init_weight_factor
        self.value_range = value_range
        
        # filters
        self.filter_size = filter_size
        self.boundary_padding_size = self.filter_size - 2
        self.n_filters = n_filters
        self.padding_mode = padding_mode
        if padding_mode == 'circular':
            self.padding_values = [None] * n_channels
        elif padding_mode == 'constant':
            self.padding_values = padding_values
        self.device = device
        self.pruning_masks = {}

        self.filters = nn.ParameterList([])
        if filters_custom_init:
            for f in filters_init_values:
                tensor = torch.tensor(f, dtype=torch.float32, device=device).view(1, 1, filter_size, filter_size)
                self.filters.append(nn.Parameter(tensor, requires_grad=learnable_filters))
        else:
            for _ in range(n_filters):
                tensor = torch.randn(1, 1, filter_size, filter_size, dtype=torch.float32, device=device)
                self.filters.append(nn.Parameter(tensor, requires_grad=learnable_filters))


    def _stochastic(self, dx: torch.Tensor):
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > self.fire_rate
        stochastic = stochastic.float().to(self.device)
        return dx * stochastic


    def _apply_filters(self, x: torch.Tensor):
        x_conv = []
        for c, x_c in enumerate(torch.split(x, 1, dim=1)):
            x_c = F.pad(x_c, tuple([self.boundary_padding_size] * 4), mode=self.padding_mode, value=self.padding_values[c])

            for i in range(self.n_filters):
                x_c_conv = F.conv2d(x_c, self.filters[i], padding=0)
                x_conv.append(x_c_conv)

        return torch.cat(x_conv, dim=1)


    def get_filters(self):
        return torch.cat([p for p in self.filters], dim=0)


    def prune_weights(self, percentile):
            """
            Prune the weights of all nn.Linear layers in the model by setting a certain percentile to zero,
            treating all nn.Linear layers as one layer for computing the quantile.
            """
            with torch.no_grad():
                # Function to recursively collect nn.Linear layer weights
                def collect_linear_weights(module, weights):
                    for child in module.children():
                        if isinstance(child, nn.Linear):
                            weights.append(child.weight.view(-1))
                        else:
                            collect_linear_weights(child, weights)

                # Collect weights from all nn.Linear layers
                linear_weights = []
                collect_linear_weights(self, linear_weights)

                all_linear_weights = torch.cat(linear_weights)

                # Compute the threshold for pruning
                threshold = torch.quantile(torch.abs(all_linear_weights), percentile / 100)

                # Apply pruning to each nn.Linear layer
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        mask = torch.abs(module.weight) > threshold
                        module.weight.data.mul_(mask)
                        self.pruning_masks[id(module)] = mask

                        
    def freeze_pruned_weights(self):
        """
        Reapply the pruning masks to freeze the pruned weights.
        """
        if self.pruning_masks:
            with torch.no_grad():
                for module in self.modules():
                    if isinstance(module, nn.Linear) and id(module) in self.pruning_masks:
                        module.weight.data.mul_(self.pruning_masks[id(module)])


    def update(self, x: torch.Tensor):
        raise NotImplementedError("Update method should be implemented in subclass")


    def forward(self, x: torch.Tensor, steps: int = 1):
        for step in range(steps):
            x = self.update(x)
            x = torch.clamp(x, min=self.value_range[0], max=self.value_range[1])
        return x



class NCA(BaseNCA):
    def __init__(self, hidden_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden = nn.Linear(self.n_filters * self.n_channels, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, self.n_channels, bias=True)
        
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.hidden.weight)
            torch.nn.init.xavier_uniform_(self.output.weight)
            
            self.hidden.weight.data = self.hidden.weight.data * self.init_weight_factor
            self.output.weight.data = self.output.weight.data * self.init_weight_factor
            self.output.bias.data.zero_()

        self.n_output_params = sum(p.numel() for p in self.output.parameters() if p.requires_grad)

        
    def update(self, x: torch.Tensor):
        # x.shape: [BATCH, WIDTH, HEIGHT, N_CHANNELS]
        x = x.transpose(1, 3)
        dx_filters = self._apply_filters(x) # dx_filters.shape: [BATCH, N_FILTERS*N_CHANNELS, HEIGHT, WIDTH]
        dx = self.hidden(dx_filters.transpose(1, 3)) # dx.shape: [BATCH, HEIGHT, WIDTH, N_HIDDEN]
        dx = F.relu(dx)
        dx = self.output(dx)
        dx = self._stochastic(dx)
        return x.transpose(1, 3) + dx
    


class DINCA(BaseNCA):
    def __init__(self, term_max_power: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.term_max_power = term_max_power
        self.n_terms = (self.term_max_power + 1) ** self.n_channels

        self.output = nn.ModuleList([nn.Linear(self.n_filters + self.n_terms, 1, bias=False) for _ in range(self.n_channels)])
        
        with torch.no_grad():
            for output_c in self.output:
                torch.nn.init.xavier_uniform_(output_c.weight)
                output_c.weight.data = output_c.weight.data * self.init_weight_factor

        self.n_output_params = sum(p.numel() for p in self.output.parameters() if p.requires_grad)
        
        
    def _compute_terms(self, x: torch.Tensor):
        x_transposed = x.transpose(0, 1)

        # Precompute powers
        powers = [torch.ones(x_transposed.shape, device=x.device)]
        for i in range(self.term_max_power):
            powers.append(powers[-1] * x_transposed)

        indices = list(itertools.product(range(self.term_max_power + 1), repeat=x_transposed.shape[0]))
        products = [torch.prod(torch.stack([powers[i][j] for j, i in enumerate(index)]), dim=0) for index in indices]

        return torch.stack(products).transpose(0, 1)


    def update(self, x: torch.Tensor):
        # x.shape: [BATCH, WIDTH, HEIGHT, N_CHANNELS]
        x = x.transpose(1, 3)

        dx_filters = self._apply_filters(x)
        # dx_filters.shape: [BATCH, N_FILTERS*N_CHANNELS, HEIGHT, WIDTH]
        dx_filters = dx_filters.view(-1, self.n_channels, self.n_filters, dx_filters.shape[2], dx_filters.shape[3])
        # dx_filters.shape: [BATCH, N_CHANNELS, N_FILTERS, HEIGHT, WIDTH]

        dx_terms = self._compute_terms(x)
        # dx_terms.shape: [BATCH, N_TERMS, HEIGHT, WIDTH]
        dx_terms_expanded = dx_terms.unsqueeze(1).repeat(1, self.n_channels, 1, 1, 1)
        # dx_terms_expanded.shape: [BATCH, N_FILTERS, N_TERMS, HEIGHT, WIDTH]

        dx = torch.cat((dx_filters, dx_terms_expanded), dim=2).transpose(2, 4)
        # dx.shape: [BATCH, N_CHANNELS, HEIGHT, WIDTH, N_FILTERS + N_TERMS]
        dx = torch.cat([self.output[c](dx[:, c, :, :, :]) for c in range(len(self.output))], dim=3)
        # dx.shape: [BATCH, HEIGHT, WIDTH, N_CHANNELS]
        dx = dx.transpose(1, 2)
        # dx.shape: [BATCH, WIDTH, HEIGHT, N_CHANNELS]

        dx = self._stochastic(dx)

        return x.transpose(1, 3) + dx
