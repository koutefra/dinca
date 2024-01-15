import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple



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

        self.filters = nn.ParameterList([])
        if filters_custom_init:
            for f in filters_init_values:
                tensor = torch.tensor(f, dtype=torch.float32, device=device).view(1, 1, filter_size, filter_size)
                self.filters.append(nn.Parameter(tensor, requires_grad=learnable_filters))
        else:
            for _ in range(n_filters):
                tensor = torch.randn(1, 1, filter_size, filter_size, dtype=torch.float32, device=device)
                self.filters.append(nn.Parameter(tensor, requires_grad=learnable_filters))


    def _stochastic(self, dx):
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > self.fire_rate
        stochastic = stochastic.float().to(self.device)
        return dx * stochastic


    def _apply_filters(self, x):
        x_conv = []
        for c, x_c in enumerate(torch.split(x, 1, dim=1)):
            x_c = F.pad(x_c, tuple([self.boundary_padding_size] * 4), mode=self.padding_mode, value=self.padding_values[c])

            for i in range(self.n_filters):
                x_c_conv = F.conv2d(x_c, self.filters[i], padding=0)
                x_conv.append(x_c_conv)

        return torch.cat(x_conv, dim=1)


    def get_filters(self):
        return torch.cat([p for p in self.filters], dim=0)


    def update(self, x):
        raise NotImplementedError("Update method should be implemented in subclass")


    def forward(self, x, steps=1):
        for step in range(steps):
            x = self.update(x)
            x = torch.clamp(x, min=self.value_range[0], max=self.value_range[1])
        return x



class NCA(BaseNCA):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden = nn.Linear(self.n_filters * self.n_channels, hidden_size, bias=True)
        self.output = nn.Linear(hidden_size, self.n_channels, bias=True)
        
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.hidden.weight)
            torch.nn.init.xavier_uniform_(self.output.weight)
            
            self.hidden.weight.data = self.hidden.weight.data * self.init_weight_factor
            self.hidden.bias.data.zero_()
            self.output.weight.data = self.output.weight.data * self.init_weight_factor
            self.output.bias.data.zero_()

        
    def update(self, x):
        # x.shape: [BATCH, WIDTH, HEIGHT, N_CHANNELS]
        x = x.transpose(1, 3)
        dx_filters = self._apply_filters(x) # dx_filters.shape: [BATCH, N_FILTERS*N_CHANNELS, HEIGHT, WIDTH]
        dx = self.hidden(dx_filters.transpose(1, 3)) # dx.shape: [BATCH, HEIGHT, WIDTH, N_HIDDEN]
        dx = F.relu(dx)
        dx = self.output(dx)
        dx = self._stochastic(dx)
        return x.transpose(1, 3) + dx
    


class PINCA(BaseNCA):
    def __init__(self, term_max_power, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert(self.n_channels == 2), "Not implemented for other number of channels than 2."

        self.term_max_power = term_max_power
        
        self.output = nn.Linear(self.n_filters * self.n_channels + self.term_max_power**2 + self.term_max_power*2, 
                                self.n_channels, bias=True)
        
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.output.weight)
            
            self.output.weight.data = self.output.weight.data * self.init_weight_factor
            self.output.bias.data.zero_()


    def _compute_terms(self, ch_1, ch_2):
        terms = []
        for i in range(1, self.term_max_power + 1):
            for j in range(1, self.term_max_power + 1):
                terms.append((ch_1 ** i) * (ch_2 ** j))

        for i in range(1, self.term_max_power + 1):
            terms.append(ch_1 ** i)
            terms.append(ch_2 ** i)
        return torch.stack(terms, dim=1)


    def update(self, x):
        # x.shape: [BATCH, WIDTH, HEIGHT, N_CHANNELS]
        x = x.transpose(1, 3)
        dx_filters = self._apply_filters(x) # dx_filters.shape: [BATCH, N_FILTERS*N_CHANNELS, HEIGHT, WIDTH]
        dx_terms = self._compute_terms(x[:, 0, :, :], x[:, 1, :, :]) # dx_terms.shape: [BATCH, N_TERMS, HEIGHT, WIDTH]
        dx = torch.cat((dx_filters, dx_terms), dim=1)
        dx = self.output(dx.transpose(1, 3)) # dx.shape: [BATCH, HEIGHT, WIDTH, N_CHANNELS]
        dx = self._stochastic(dx)
        return x.transpose(1, 3) + dx