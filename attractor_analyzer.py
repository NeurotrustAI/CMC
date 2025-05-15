"""
Attractor analysis tools for analyzing phase space dynamics of neural ODE models.
Part of the visualization module.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torchdiffeq import odeint


class AttractorAnalyzer:
    """Analyze and visualize attractors in FlexibleCMC phase space"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def extract_attractor_states(self, dataloader, num_samples_per_class=50):
        """Extract final states (attractors) for each digit class"""
        attractors = {i: [] for i in range(10)}
        initial_states = {i: [] for i in range(10)}
        counts = {i: 0 for i in range(10)}

        with torch.no_grad():
            for data, targets in dataloader:
                data = data.to(self.device)

                # Get initial and final states
                batch_size = data.shape[0]

                # Use the model's forward method to get trajectories
                for i in range(batch_size):
                    label = targets[i].item()
                    if counts[label] < num_samples_per_class:
                        _, trajectory = self.model.forward(
                            data[i:i + 1],
                            return_trajectory=True
                        )

                        if trajectory is not None:
                            initial_state = trajectory[0].squeeze()
                            final_state = trajectory[-1].squeeze()

                            initial_states[label].append(initial_state.cpu().numpy())
                            attractors[label].append(final_state.cpu().numpy())
                            counts[label] += 1

                # Check if we have enough samples
                if all(count >= num_samples_per_class for count in counts.values()):
                    break

        # Convert to numpy arrays
        for label in range(10):
            attractors[label] = np.array(attractors[label])
            initial_states[label] = np.array(initial_states[label])

        return attractors, initial_states

    def compute_attractor_properties(self, attractors):
        """Compute statistical properties of attractors"""
        properties = {}

        for label in range(10):
            if len(attractors[label]) > 0:
                states = attractors[label]

                # Compute mean attractor
                mean_state = states.mean(axis=0)

                # Compute covariance and eigenvalues (stability analysis)
                if len(states) > 1:
                    cov_matrix = np.cov(states.T)
                    try:
                        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                        eigenvalues = np.real(eigenvalues)  # Take real part
                    except:
                        eigenvalues = np.zeros(states.shape[1])
                        eigenvectors = np.eye(states.shape[1])
                else:
                    eigenvalues = np.zeros(states.shape[1])
                    eigenvectors = np.eye(states.shape[1])

                # Compute basin size (variance)
                basin_size = states.var(axis=0).mean()

                # Compute separability from other attractors
                separability = []
                for other_label in range(10):
                    if other_label != label and len(attractors[other_label]) > 0:
                        other_mean = attractors[other_label].mean(axis=0)
                        distance = np.linalg.norm(mean_state - other_mean)
                        separability.append(distance)

                properties[label] = {
                    'mean_state': mean_state,
                    'eigenvalues': eigenvalues,
                    'eigenvectors': eigenvectors,
                    'basin_size': basin_size,
                    'mean_separability': np.mean(separability) if separability else 0
                }

        return properties

    def visualize_attractor_landscape(self, attractors, method='pca', save_path=None):
        """Visualize the attractor landscape using dimensionality reduction"""
        # Combine all attractors
        all_states = []
        labels = []

        for label in range(10):
            if len(attractors[label]) > 0:
                all_states.extend(attractors[label])
                labels.extend([label] * len(attractors[label]))

        if not all_states:
            print("No attractor states to visualize")
            return None

        all_states = np.array(all_states)
        labels = np.array(labels)

        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=3)
            reduced_states = reducer.fit_transform(all_states)
        elif method == 'tsne':
            reducer = TSNE(n_components=3, perplexity=30)
            reduced_states = reducer.fit_transform(all_states)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        # Create interactive 3D plot
        fig = go.Figure()

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for label in range(10):
            mask = labels == label
            if mask.any():
                fig.add_trace(go.Scatter3d(
                    x=reduced_states[mask, 0],
                    y=reduced_states[mask, 1],
                    z=reduced_states[mask, 2],
                    mode='markers',
                    name=f'Digit {label}',
                    marker=dict(
                        size=5,
                        color=f'rgb({colors[label][0] * 255:.0f},{colors[label][1] * 255:.0f},{colors[label][2] * 255:.0f})',
                        opacity=0.8
                    )
                ))

                # Add attractor center
                center = reduced_states[mask].mean(axis=0)
                fig.add_trace(go.Scatter3d(
                    x=[center[0]],
                    y=[center[1]],
                    z=[center[2]],
                    mode='markers',
                    name=f'Center {label}',
                    marker=dict(
                        size=15,
                        color=f'rgb({colors[label][0] * 255:.0f},{colors[label][1] * 255:.0f},{colors[label][2] * 255:.0f})',
                        symbol='diamond'
                    ),
                    showlegend=False
                ))

        fig.update_layout(
            title=f'Attractor Landscape ({method.upper()})',
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            width=900,
            height=700
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_attractor_stability(self, properties, save_path=None):
        """Plot stability analysis of attractors"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Basin sizes
        labels = list(properties.keys())
        basin_sizes = [properties[l]['basin_size'] for l in labels]
        axes[0, 0].bar(labels, basin_sizes)
        axes[0, 0].set_title('Basin Sizes')
        axes[0, 0].set_xlabel('Digit')
        axes[0, 0].set_ylabel('Mean Variance')
        axes[0, 0].grid(True, alpha=0.3)

        # Separability
        separabilities = [properties[l]['mean_separability'] for l in labels]
        axes[0, 1].bar(labels, separabilities)
        axes[0, 1].set_title('Mean Separability')
        axes[0, 1].set_xlabel('Digit')
        axes[0, 1].set_ylabel('Mean Distance to Others')
        axes[0, 1].grid(True, alpha=0.3)

        # Eigenvalue spectra
        for label in labels:
            eigenvalues = np.sort(properties[label]['eigenvalues'])[::-1][:20]
            axes[1, 0].plot(eigenvalues, label=f'Digit {label}', alpha=0.7)
        axes[1, 0].set_title('Eigenvalue Spectra (Top 20)')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('Eigenvalue')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Stability heatmap
        stability_matrix = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                if i != j and i in properties and j in properties:
                    dist = np.linalg.norm(
                        properties[i]['mean_state'] - properties[j]['mean_state']
                    )
                    stability_matrix[i, j] = dist

        sns.heatmap(stability_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                    ax=axes[1, 1], cbar_kws={'label': 'Distance'})
        axes[1, 1].set_title('Pairwise Attractor Distances')
        axes[1, 1].set_xlabel('Digit')
        axes[1, 1].set_ylabel('Digit')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def analyze_convergence_dynamics(self, dataloader, num_samples=10, save_path=None):
        """Analyze how different digits converge to their attractors"""
        convergence_data = {i: [] for i in range(10)}

        with torch.no_grad():
            sample_count = {i: 0 for i in range(10)}

            for data, targets in dataloader:
                data = data.to(self.device)

                for idx, target in enumerate(targets):
                    label = target.item()
                    if sample_count[label] >= num_samples:
                        continue

                    # Get trajectory
                    _, trajectory = self.model.forward(
                        data[idx:idx + 1],
                        return_trajectory=True
                    )

                    if trajectory is not None:
                        convergence_data[label].append(
                            trajectory.squeeze().cpu().numpy()
                        )
                        sample_count[label] += 1

                if all(count >= num_samples for count in sample_count.values()):
                    break

        # Plot convergence
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()

        for label in range(10):
            ax = axes[label]

            if convergence_data[label]:
                trajectories = np.array(convergence_data[label])

                # Compute distance to final state over time
                final_states = trajectories[:, -1, :]
                mean_final = final_states.mean(axis=0)

                distances = []
                for t in range(trajectories.shape[1]):
                    states_t = trajectories[:, t, :]
                    dist_t = np.mean([
                        np.linalg.norm(state - mean_final)
                        for state in states_t
                    ])
                    distances.append(dist_t)

                time_points = np.linspace(0, 1, len(distances))
                ax.plot(time_points, distances, linewidth=2)
                ax.set_title(f'Digit {label}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Distance to Attractor')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_phase_portrait(self, attractors, dimensions=(0, 1), save_path=None):
        """Create a 2D phase portrait of the attractors"""
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for label in range(10):
            if len(attractors[label]) > 0:
                states = attractors[label]
                ax.scatter(states[:, dimensions[0]], states[:, dimensions[1]],
                           color=colors[label], label=f'Digit {label}',
                           alpha=0.6, s=30)

                # Add mean point
                mean_state = states.mean(axis=0)
                ax.scatter(mean_state[dimensions[0]], mean_state[dimensions[1]],
                           color=colors[label], s=200, marker='*',
                           edgecolor='black', linewidth=2)

        ax.set_xlabel(f'Dimension {dimensions[0]}')
        ax.set_ylabel(f'Dimension {dimensions[1]}')
        ax.set_title('Phase Portrait of Attractors')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig