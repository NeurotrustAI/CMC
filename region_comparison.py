import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .loss_surface import LossLandscapeVisualizer
from .neural_dynamics import NeuralDynamicsVisualizer


class RegionComparisonVisualizer:
    def __init__(self, base_path, model_loader, data_loader):
        self.base_path = Path(base_path)
        self.model_loader = model_loader
        self.data_loader = data_loader

    def plot_accuracy_comparison(self, results_dir="results/region_comparison", save_path=None):
        """Enhanced accuracy bar plot with error bars"""
        plt.figure(figsize=(12, 6))

        accuracies = []
        errors = []
        n_regions_list = range(1, 6)

        # Load results
        for n_regions in n_regions_list:
            result_file = np.load(f"{results_dir}/mnist_{n_regions}regions_results.npy",
                                  allow_pickle=True).item()
            accuracies.append(result_file['accuracies'][0])
            errors.append(result_file['errors'][0])

        # Create bar plot with enhanced styling
        bars = plt.bar(n_regions_list, accuracies, yerr=errors,
                       capsize=5, color='skyblue', alpha=0.7)

        # Add value labels on top of bars
        for bar, acc, err in zip(bars, accuracies, errors):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=12)

        plt.xlabel('Number of Regions', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Model Accuracy vs Number of Regions', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.xticks(n_regions_list, fontsize=12)
        plt.yticks(fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def compare_loss_landscapes(self, save_dir=None):
        """Compare loss landscapes across different numbers of regions"""
        n_regions_list = range(1, 6)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), subplot_kw={'projection': '3d'})

        for idx, n_regions in enumerate(n_regions_list):
            # Load model
            model = self.model_loader.load_model(n_regions=n_regions)
            visualizer = LossLandscapeVisualizer(model,
                                                 criterion=torch.nn.CrossEntropyLoss(),
                                                 train_loader=self.data_loader.train_loader)

            # Compute and plot loss surface
            X, Y, Z = visualizer.compute_loss_surface()
            Z = (Z - Z.min()) / (Z.max() - Z.min())

            ax = axes[idx]
            surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='black',
                                   linewidth=0.5, alpha=0.8)

            ax.set_title(f'{n_regions} Region(s)')
            ax.view_init(elev=30, azim=45)

        # Add common labels
        fig.text(0.5, 0, 'Direction 1', ha='center', fontsize=14)
        fig.text(0.08, 0.5, 'Direction 2', va='center', rotation='vertical', fontsize=14)
        fig.suptitle('Loss Landscape Comparison Across Region Counts', fontsize=16)

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/loss_landscape_comparison.png",
                        dpi=300, bbox_inches='tight')
        plt.show()

    def compare_dynamics(self, sample_input, save_dir=None):
        """Compare neural dynamics across different numbers of regions"""
        n_regions_list = range(1, 6)
        fig, axes = plt.subplots(len(n_regions_list), 1, figsize=(15, 4 * len(n_regions_list)))

        for idx, n_regions in enumerate(n_regions_list):
            # Load model
            model = self.model_loader.load_model(n_regions=n_regions)
            visualizer = NeuralDynamicsVisualizer(model)

            # Get dynamics
            dynamics, time = visualizer.get_dynamics(sample_input)

            # Plot dynamics
            ax = axes[idx]
            voltage_indices = [0, 2, 4, 6]
            for v_idx in voltage_indices:
                ax.plot(time, dynamics[:, v_idx],
                        label=visualizer.population_names[v_idx])

            ax.set_title(f'{n_regions} Region(s)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=14)
        fig.text(0.04, 0.5, 'Voltage', va='center', rotation='vertical', fontsize=14)
        fig.suptitle('Neural Dynamics Comparison Across Region Counts', fontsize=16)

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/dynamics_comparison.png",
                        dpi=300, bbox_inches='tight')
        plt.show()