import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchdiffeq import odeint


class NeuralDynamicsVisualizer:
    def __init__(self, model):
        self.model = model
        self.population_names = [
            'Spiny Stellate (V)', 'Spiny Stellate (G)',
            'Superficial Pyramidal (V)', 'Superficial Pyramidal (G)',
            'Inhibitory Interneurons (V)', 'Inhibitory Interneurons (G)',
            'Deep Pyramidal (V)', 'Deep Pyramidal (G)'
        ]

    def get_dynamics(self, input_data, t_span=[0, 1], dt=0.01):
        """Compute neural dynamics for given input"""
        self.model.eval()
        with torch.no_grad():
            # Get initial state
            x0 = self.model.input_network(input_data)

            # Create time points
            t = torch.linspace(t_span[0], t_span[1], 100).to(x0.device)

            # Solve ODE using the same settings as training
            solution = odeint(
                self.model.node,
                x0,
                t,
                method='euler',
                options={'step_size': 0.1}
            )

        return solution.cpu().numpy(), t.cpu().numpy()

    def plot_voltage_dynamics(self, dynamics, time, save_path=None):
        plt.figure(figsize=(15, 10))

        # Plot voltage traces (odd indices are voltage)
        voltage_indices = [0, 2, 4, 6]
        colors = sns.color_palette("husl", n_colors=len(voltage_indices))

        # Ensure dynamics has the right shape
        if len(dynamics.shape) == 3:  # If batched
            dynamics = dynamics.squeeze(1)  # Remove batch dimension

        for idx, color in zip(voltage_indices, colors):
            plt.plot(time, dynamics[:, idx],
                     label=self.population_names[idx],
                     color=color, linewidth=2)

        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Voltage', fontsize=12)
        plt.title('Neural Population Voltage Dynamics', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_phase_portrait(self, dynamics, population_idx=0, save_path=None):
        plt.figure(figsize=(10, 10))

        # Ensure dynamics has the right shape
        if len(dynamics.shape) == 3:  # If batched
            dynamics = dynamics.squeeze(1)

        # Get voltage and conductance for the selected population
        v_idx = population_idx * 2
        g_idx = v_idx + 1

        plt.plot(dynamics[:, v_idx], dynamics[:, g_idx], 'b-', linewidth=1, alpha=0.6)
        plt.plot(dynamics[0, v_idx], dynamics[0, g_idx], 'go', label='Start')
        plt.plot(dynamics[-1, v_idx], dynamics[-1, g_idx], 'ro', label='End')

        plt.xlabel(f'{self.population_names[v_idx]} Voltage', fontsize=12)
        plt.ylabel(f'{self.population_names[v_idx]} Conductance', fontsize=12)
        plt.title(f'Phase Portrait: {self.population_names[v_idx]}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()