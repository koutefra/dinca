from .constraints import K2M
import torch
from typing import Callable



class BaseLoss:
    def __init__(self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], padding_size: int, channel_mode: str):
        self.loss_fn = loss_fn
        self.p = padding_size

        if channel_mode == 'gray':
            self.channel_slice = slice(0, 1)
        elif channel_mode == 'rgb':
            self.channel_slice = slice(0, 3)
        else:
            assert f"No such channel loss mode implemented: '{channel_mode}'"


    def _get_loss(self, x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        idx = slice(self.p, -self.p if self.p != 0 else None)
        n_batches = x.shape[0]
        loss = [self.loss_fn(x[b, idx, idx, self.channel_slice], x_ref[:, :, self.channel_slice]) for b in range(n_batches)]
        return sum(loss) / n_batches


    def compute(self, model, x_outs, x_refs):
        raise NotImplementedError("This method should be implemented by subclasses")
    


class WeightedPixelLoss(BaseLoss):
    def __init__(self, reg_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_factor = reg_factor
        
    def compute(self, model, x_outs, x_refs):
        loss_pixel = sum([self._get_loss(x_out, x_ref) for x_out, x_ref in zip(x_outs, x_refs)]) / len(x_refs)
        loss_reg = sum(p.abs().sum() for p in model.output.parameters() if p.requires_grad) / model.n_output_params
        loss_reg_weighted = loss_reg * self.reg_factor
        total_loss = loss_pixel + loss_reg_weighted

        loss_elements = {'raw_loss_pixel': loss_pixel.item(), 'raw_loss_reg': loss_reg.item(),
                         'loss_pixel': loss_pixel.item(), 'loss_reg': loss_reg_weighted.item()}
        
        return total_loss, loss_elements


    
class FilterAwareWeightedPixelLoss(BaseLoss):
    def __init__(self, reg_factor: float, prescribed_moments: torch.Tensor, filters_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_factor = reg_factor
        self.prescribed_moments = prescribed_moments
        self.filters_factor = filters_factor


    def _get_moments(self, filters: torch.Tensor):
        filter_size = filters.shape[2]
        k2m = K2M([filter_size, filter_size]).to(filters.device)
        moments = k2m(filters.double())
        moments = moments.float()
        return moments / len(filters)


    def compute(self, model, x_outs, x_refs):
        loss_pixel = sum([self._get_loss(x_out, x_ref) for x_out, x_ref in zip(x_outs, x_refs)]) / len(x_refs)
        loss_reg = sum(p.abs().sum() for p in model.output.parameters() if p.requires_grad) / model.n_output_params
        loss_reg_weighted = loss_reg * self.reg_factor

        moments = self._get_moments(model.get_filters())[:, 0, :, :]
        loss_filters = self.loss_fn(moments, self.prescribed_moments)
        loss_filters_weighted = loss_filters * self.filters_factor

        total_loss = loss_pixel + loss_reg_weighted + loss_filters_weighted

        loss_elements = {'raw_loss_pixel': loss_pixel.item(), 'raw_loss_reg': loss_reg.item(),
                         'raw_loss_filters': loss_filters.item(), 'loss_pixel': loss_pixel.item(),
                         'loss_reg': loss_reg_weighted.item(), 'loss_filters': loss_filters_weighted.item()}
        
        return total_loss, loss_elements