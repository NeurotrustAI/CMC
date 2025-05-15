import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from torchdiffeq import odeint
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors


class NeuralActivityAnalyzer:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.device = next(model.parameters()).device
        self.population_names = [
            'Spiny Stellate (V)', 'Spiny Stellate (G)',
            'Superficial Pyramidal (V)', 'Superficial Pyramidal (G)',
            'Inhibitory Interneurons (V)', 'Inhibitory Interneurons (G)',
            'Deep Pyramidal (V)', 'Deep Pyramidal (G)'
        ]

    # ... [previous methods remain the same] ...

    def plot_population_activity_heatmap(self, save_path=None):
        """Create heatmap of neural population activities for different digits"""
        trajectories = self.get_digit_trajectories(n_samples_per_digit=5)

        # Compute average activities
        avg_activities = np.zeros((10, 8))  # 10 digits, 8 neural states

        for digit in range(10):
            digit_trajectories = trajectories[digit]
            # Average over samples and time
            avg_activities[digit] = np.mean([np.mean(traj, axis=0) for traj in digit_trajectories], axis=0).squeeze()

        plt.figure(figsize=(15, 8))
        sns.heatmap(avg_activities,
                    xticklabels=['SS-V', 'SS-G', 'SP-V', 'SP-G', 'II-V', 'II-G', 'DP-V', 'DP-G'],
                    yticklabels=range(10),
                    cmap='viridis',
                    center=0,
                    annot=True,
                    fmt='.2f',
                    cbar_kws={'label': 'Average Activity'})

        plt.title('Neural Population Activity by Digit', pad=20, size=16)
        plt.xlabel('Neural Population', fontsize=12)
        plt.ylabel('Digit', fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_digit_comparison(self, digit1, digit2, save_path=None):
        """Compare neural trajectories between two digits"""
        trajectories = self.get_digit_trajectories(n_samples_per_digit=3)

        fig = plt.figure(figsize=(20, 5))

        # Voltage components subplot
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection='3d')

        colors = ['blue', 'red']
        digits = [digit1, digit2]

        for i, digit in enumerate(digits):
            digit_trajectories = trajectories[digit]
            for traj in digit_trajectories:
                # Plot SS voltage
                ax1.plot(traj[:, 0, 0], color=colors[i], alpha=0.6,
                         label=f'Digit {digit}' if traj is digit_trajectories[0] else "")

                # Plot SP voltage
                ax2.plot(traj[:, 0, 2], color=colors[i], alpha=0.6,
                         label=f'Digit {digit}' if traj is digit_trajectories[0] else "")

                # 3D trajectory
                ax3.plot3D(traj[:, 0, 0], traj[:, 0, 2], traj[:, 0, 4],
                           color=colors[i], alpha=0.6,
                           label=f'Digit {digit}' if traj is digit_trajectories[0] else "")

        ax1.set_title('Spiny Stellate Voltage')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Voltage')
        ax1.legend()

        ax2.set_title('Superficial Pyramidal Voltage')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Voltage')
        ax2.legend()

        ax3.set_title('3D Neural Trajectory')
        ax3.set_xlabel('SS-V')
        ax3.set_ylabel('SP-V')
        ax3.set_zlabel('II-V')
        ax3.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_phase_space_density(self, save_path=None):
        """Create a density plot of neural trajectories in phase space"""
        trajectories = self.get_digit_trajectories(n_samples_per_digit=5)

        plt.figure(figsize=(15, 15))

        # Combine all trajectories
        all_ss_v = []
        all_sp_v = []
        colors = []

        for digit in range(10):
            digit_trajectories = trajectories[digit]
            for traj in digit_trajectories:
                all_ss_v.extend(traj[:, 0, 0])
                all_sp_v.extend(traj[:, 0, 2])
                colors.extend([digit] * len(traj))

        # Create density plot
        sns.kdeplot(x=all_ss_v, y=all_sp_v, cmap='viridis', fill=True, levels=20)

        # Overlay scatter plot with low alpha
        plt.scatter(all_ss_v, all_sp_v, c=colors, cmap='tab10', alpha=0.1, s=1)

        plt.title('Neural Phase Space Density', pad=20, size=16)
        plt.xlabel('Spiny Stellate Voltage', fontsize=12)
        plt.ylabel('Superficial Pyramidal Voltage', fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def create_interactive_plot(self, digit, save_path=None):
        """Create an interactive visualization showing evolving neural states"""
        trajectories = self.get_digit_trajectories(n_samples_per_digit=1)
        traj = trajectories[digit][0]

        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 2)

        # Main 3D trajectory
        ax1 = fig.add_subplot(gs[:, 0], projection='3d')
        # Voltage traces
        ax2 = fig.add_subplot(gs[0, 1])
        # Phase portrait
        ax3 = fig.add_subplot(gs[1, 1])

        def update(frame):
            ax1.clear()
            ax2.clear()
            ax3.clear()

            # 3D trajectory
            ax1.plot3D(traj[:frame, 0, 0], traj[:frame, 0, 2], traj[:frame, 0, 4],
                       'b-', label=f'Digit {digit}')
            ax1.set_title('Neural Trajectory')
            ax1.set_xlabel('SS-V')
            ax1.set_ylabel('SP-V')
            ax1.set_zlabel('II-V')

            # Voltage traces
            time = np.arange(frame)
            ax2.plot(time, traj[:frame, 0, 0], 'b-', label='SS-V')
            ax2.plot(time, traj[:frame, 0, 2], 'r-', label='SP-V')
            ax2.plot(time, traj[:frame, 0, 4], 'g-', label='II-V')
            ax2.set_title('Voltage Traces')
            ax2.legend()

            # Phase portrait
            ax3.plot(traj[:frame, 0, 0], traj[:frame, 0, 2], 'b-')
            ax3.set_title('Phase Portrait')
            ax3.set_xlabel('SS-V')
            ax3.set_ylabel('SP-V')

            return ax1, ax2, ax3

        anim = FuncAnimation(fig, update, frames=len(traj), interval=50, blit=True)

        if save_path:
            anim.save(save_path, writer='pillow', fps=30)

        plt.close()
        print(f"Animation saved to {save_path}")
        return anim