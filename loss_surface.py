# Visualize Loss Surface of CMC Model
# PKD Dec 24

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib import cm


class LossLandscapeVisualizer:
    def __init__(self, model, criterion, train_loader, device='cuda'):
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device

    def compute_loss_surface(self, param1_range=(-1, 1), param2_range=(-1, 1), n_points=50):
        """Higher resolution for smoother plots"""
        params = torch.nn.utils.parameters_to_vector(self.model.parameters())

        x = np.linspace(param1_range[0], param1_range[1], n_points)
        y = np.linspace(param2_range[0], param2_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        U = torch.randn_like(params)
        V = torch.randn_like(params)
        U = U / torch.norm(U)
        V = V - (V @ U) * U
        V = V / torch.norm(V)

        original_params = params.clone()

        print("Computing loss landscape...")
        for i in range(n_points):
            for j in range(n_points):
                new_params = original_params + X[i, j] * U + Y[i, j] * V
                torch.nn.utils.vector_to_parameters(new_params, self.model.parameters())

                batch_loss = 0
                n_batches = 0
                with torch.no_grad():
                    for data, target in self.train_loader:
                        data = data.to(self.device)
                        target = target.to(self.device)
                        output = self.model(data)
                        batch_loss += self.criterion(output, target).item()
                        n_batches += 1
                        if n_batches >= 5:  # Sample fewer batches for speed
                            break
                Z[i, j] = batch_loss / n_batches

        # Restore original parameters
        torch.nn.utils.vector_to_parameters(original_params, self.model.parameters())

        return X, Y, Z

    def plot_dramatic_loss_surface(self, save_path=None, colormap='ocean'):
        """Create a more dramatic 3D visualization with custom colormap"""
        X, Y, Z = self.compute_loss_surface()

        # Normalize Z for better visualization
        Z = (Z - Z.min()) / (Z.max() - Z.min())

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Select colormap based on preference
        if colormap == 'ocean':
            cmap = cm.get_cmap('ocean')
        elif colormap == 'nipy_spectral':
            cmap = cm.get_cmap('nipy_spectral')
        elif colormap == 'twilight':
            cmap = cm.get_cmap('twilight')
        elif colormap == 'plasma':
            cmap = cm.get_cmap('plasma')
        else:
            # Custom colormap with ocean-like colors
            colors_list = ['#000033', '#000055', '#0000ff', '#0055ff', '#00ffff', '#55ff00', '#ffff00', '#ff5500',
                           '#ff0000']
            cmap = colors.LinearSegmentedColormap.from_list('custom_ocean', colors_list)

        # Plot surface with enhanced visual style
        surf = ax.plot_surface(X, Y, Z,
                               cmap=cmap,
                               edgecolor='black',
                               linewidth=0.3,
                               alpha=0.9,
                               antialiased=True,
                               shade=True)

        # Add contour projection on bottom plane
        offset = Z.min() - 0.1
        cset = ax.contour(X, Y, Z, zdir='z', offset=offset, cmap='twilight', alpha=0.5, levels=10)

        # Enhanced lighting for more dramatic effect
        ax.set_box_aspect([1, 1, 0.8])

        # Customize the plot
        ax.view_init(elev=30, azim=45)
        ax.set_xlabel('Direction 1', labelpad=20, fontsize=14)
        ax.set_ylabel('Direction 2', labelpad=20, fontsize=14)
        ax.set_zlabel('Loss', labelpad=20, fontsize=14)
        ax.set_title('CMC Neural ODE Loss Landscape', pad=20, size=18, weight='bold')

        # Add colorbar with nice formatting
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        cbar.set_label('Normalized Loss', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # Set background color
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Grid styling
        ax.grid(True, alpha=0.2, linestyle='--')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()
        return fig, ax

    def create_rotating_animation(self, save_path=None, n_frames=60, fps=30, colormap='ocean'):
        """Create smooth rotating animation of loss landscape"""
        X, Y, Z = self.compute_loss_surface()
        Z = (Z - Z.min()) / (Z.max() - Z.min())

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Select colormap
        if colormap == 'ocean':
            cmap = cm.get_cmap('ocean')
        elif colormap == 'nipy_spectral':
            cmap = cm.get_cmap('nipy_spectral')
        else:
            cmap = cm.get_cmap('twilight')

        def update(frame):
            ax.clear()

            surf = ax.plot_surface(X, Y, Z,
                                   cmap=cmap,
                                   edgecolor='black',
                                   linewidth=0.3,
                                   alpha=0.9)

            offset = Z.min() - 0.1
            ax.contour(X, Y, Z, zdir='z', offset=offset, cmap='twilight', alpha=0.5)

            ax.set_xlabel('Direction 1', labelpad=20, fontsize=14)
            ax.set_ylabel('Direction 2', labelpad=20, fontsize=14)
            ax.set_zlabel('Loss', labelpad=20, fontsize=14)
            ax.set_title('CMC Neural ODE Loss Landscape', pad=20, size=18, weight='bold')

            # Update view angle
            ax.view_init(elev=30, azim=frame * (360 / n_frames))

            return [surf]

        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)

        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)

        plt.close()
        print(f"Animation saved to {save_path}")
        return anim

    def plot_loss_contour_dramatic(self, save_path=None, colormap='ocean'):
        """Create enhanced 2D contour plot"""
        X, Y, Z = self.compute_loss_surface()
        Z = (Z - Z.min()) / (Z.max() - Z.min())

        plt.figure(figsize=(15, 12))

        # Select colormap
        if colormap == 'ocean':
            cmap = 'ocean'
        elif colormap == 'nipy_spectral':
            cmap = 'nipy_spectral'
        elif colormap == 'twilight':
            cmap = 'twilight'
        else:
            cmap = 'plasma'

        # Create dramatic contour plot
        levels = 50
        contour_filled = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
        contour_lines = plt.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)

        # Add colorbar
        cbar = plt.colorbar(contour_filled, label='Normalized Loss')
        cbar.ax.tick_params(labelsize=10)

        # Customize plot
        plt.xlabel('Direction 1', fontsize=14)
        plt.ylabel('Direction 2', fontsize=14)
        plt.title('CMC Loss Landscape Contour View', fontsize=18, pad=20, weight='bold')

        # Add grid with custom style
        plt.grid(True, alpha=0.2, linestyle='--')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()

    def plot_multiple_views(self, save_path=None, colormap='ocean'):
        """Create multiple views of the loss landscape in one figure"""
        X, Y, Z = self.compute_loss_surface()
        Z = (Z - Z.min()) / (Z.max() - Z.min())

        fig = plt.figure(figsize=(20, 15))

        # Select colormap
        if colormap == 'ocean':
            cmap = cm.get_cmap('ocean')
        elif colormap == 'nipy_spectral':
            cmap = cm.get_cmap('nipy_spectral')
        else:
            cmap = cm.get_cmap('twilight')

        # 3D view 1
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9, antialiased=True)
        ax1.view_init(elev=30, azim=45)
        ax1.set_title('View 1: Standard', fontsize=14)

        # 3D view 2
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9, antialiased=True)
        ax2.view_init(elev=60, azim=135)
        ax2.set_title('View 2: Top-down', fontsize=14)

        # Contour plot
        ax3 = fig.add_subplot(2, 2, 3)
        contour = ax3.contourf(X, Y, Z, levels=30, cmap=cmap)
        ax3.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)
        ax3.set_title('Contour View', fontsize=14)
        ax3.set_xlabel('Direction 1')
        ax3.set_ylabel('Direction 2')

        # 1D slices
        ax4 = fig.add_subplot(2, 2, 4)
        mid_idx = len(X) // 2
        ax4.plot(X[mid_idx, :], Z[mid_idx, :], 'b-', linewidth=2, label='Slice along Direction 1')
        ax4.plot(Y[:, mid_idx], Z[:, mid_idx], 'r-', linewidth=2, label='Slice along Direction 2')
        ax4.set_title('1D Loss Curves', fontsize=14)
        ax4.set_xlabel('Parameter Value')
        ax4.set_ylabel('Normalized Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('CMC Loss Landscape Analysis', fontsize=20, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()
        return fig