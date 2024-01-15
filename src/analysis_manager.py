from typing import Any, Dict, List
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns



class AnalysisManager:
    def __init__(self, figures_folder_path: str):
        self.figures_folder_path = figures_folder_path


    def get_loss_plot(self, loss_history):
        assert (len(loss_history) != 0), "The loss history cannot be plotted since there are no loss data."

        weighted_loss_values = {key: [] for key in loss_history[0].keys() if 'loss_' in key and 'raw_loss_' not in key}
        raw_loss_values = {key: [] for key in loss_history[0].keys() if 'raw_loss_' in key}

        # Aggregate data
        for loss_dict in loss_history:
            for key in weighted_loss_values:
                value = loss_dict.get(key, 1e-10)
                weighted_loss_values[key].append(0 if value <= 1e-10 else np.log(value))
            for key in raw_loss_values:
                value = loss_dict.get(key, 1e-10)
                raw_loss_values[key].append(0 if value <= 1e-10 else np.log(value)) # 1e-10 is a small constant

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # Two subplots
        epochs = range(1, len(loss_history) + 1)

        for key, values in weighted_loss_values.items():
            axs[0].plot(epochs, values, label=key)
        for key, values in raw_loss_values.items():
            axs[1].plot(epochs, values, label=key)

        axs[0].set_title("Weighted Loss Metrics Over Epochs")
        axs[1].set_title("Raw Loss Metrics Over Epochs")
        for ax in axs:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Log Loss Value")
            ax.legend()

        plot_path = os.path.join(self.figures_folder_path, 'loss_plot.png')
        fig.savefig(plot_path)
        plt.close(fig)


    def get_simulation_plot(self, model: torch.nn.Module, x_init_batch: torch.Tensor, n_steps: int):
        x_outs = [x_init_batch[0:1, :, :, :]]
        for step in range(n_steps):
            x_out = model(x_outs[-1])[0:1, :, :, :] # take the first batch
            x_outs.append(x_out)

        x_outs_tensor = torch.cat(x_outs, dim=0)[:, :, :, 0:1] # take only the first image
        x_outs_tensor = x_outs_tensor.permute(0, 3, 1, 2)

        # Reshape and permute tensor to [n_steps * batch_size, n_channels, width, height]
        grid = torchvision.utils.make_grid(x_outs_tensor, ncols=int(np.sqrt(len(x_outs))))

        # Convert to PIL Image for saving
        np_img = grid.detach().cpu().numpy().transpose(1, 2, 0)
        pil_img = Image.fromarray((np_img * 255).astype(np.uint8))

        plot_path = os.path.join(self.figures_folder_path, 'simulation.png')
        pil_img.save(plot_path)


    def get_batch_plot(self, x_outs: List[torch.Tensor], x_init: torch.Tensor, x_refs: List[torch.Tensor], mode: str):
        # x_outs[i] is of shape [batch_size, width, height, n_channels]
        # x_init is of shape [width, height, n_channels]
        # x_refs[i] is oh shape [width, height, n_channels]

        emerged_images = [x_out.detach().cpu().numpy() for x_out in x_outs]  # List of [batch_size, width, height, n_channels]
        init_image = x_init.detach().cpu().numpy()  # Shape: [width, height, n_channels]
        predefined_images = [init_image] + [x_ref.detach().cpu().numpy() for x_ref in x_refs]  # List of [width, height, n_channels]

        batch_size = emerged_images[0].shape[0]
        n_ref_states = len(predefined_images)

        fig, axs = plt.subplots(batch_size + 1, n_ref_states, figsize=[15, 8])

        # Plot emerged images
        for i in range(batch_size):
            for j in range(n_ref_states):
                img = emerged_images[j][i, :, :, 0] if mode == 'gray' else emerged_images[j][i, :, :, 0:3]
                axs[i, j].imshow(img, cmap=mode if mode == 'gray' else None)
                axs[i, j].axis('off')
                axs[i, j].set_title(f'Batch {i + 1}, Output {j}' if j > 0 else f'Batch {i + 1}, Init')

        # Plot predefined images
        for j, ref_img in enumerate(predefined_images):
            ref_img = ref_img[:, :, 0] if mode == 'gray' else ref_img[:, :, 0:3]
            axs[batch_size, j].imshow(ref_img, cmap=mode if mode == 'gray' else None)
            axs[batch_size, j].axis('off')
            axs[batch_size, j].set_title(f'Ref {j + 1}')

        plt.tight_layout()

        plot_path = os.path.join(self.figures_folder_path, 'batch_plot.png')
        plt.savefig(plot_path)
        plt.close()


    def get_filter_values_plot(self, filters: torch.Tensor):
        """
        This function takes a tensor of filters and creates a subplot containing a grid
        of all filters visualized as heatmaps on the same scale.

        Parameters:
        filters (torch.Tensor): Tensor of shape [n_filters, 1, filter_size, filter_size]
        """
        n_filters = filters.shape[0]
        grid_size = int(n_filters ** 0.5)
        if grid_size * grid_size < n_filters:
            grid_size += 1

        # Find the global minimum and maximum across all filters for consistent scaling
        min_val = filters.min().item()
        max_val = filters.max().item()
        # Ensure zero is centered with a neutral color
        vmax = max(abs(min_val), abs(max_val))
        vmin = -vmax

        plt.figure(figsize=(grid_size * 2, grid_size * 2))
        for i in range(n_filters):
            ax = plt.subplot(grid_size, grid_size, i + 1)
            filter_values = filters[i, 0, :, :].cpu().detach().numpy()
            sns.heatmap(filter_values, square=True, cmap='coolwarm', cbar=False, ax=ax, vmin=vmin, vmax=vmax)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Add an axes to the right side of the figure for the colorbar
        fig = plt.gcf()
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        plt.colorbar(ax.collections[0], cax=cbar_ax)

        plot_path = os.path.join(self.figures_folder_path, 'filters_plot.png')
        plt.savefig(plot_path)
        plt.close()