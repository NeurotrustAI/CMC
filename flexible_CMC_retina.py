"""
Flexible CMC model with retinal preprocessing layer
Can be used for various vision tasks (MNIST, CIFAR, ImageNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from .flexible_CMC import FlexibleCMC


class RetinalConvLayer(nn.Module):
    """Biologically-inspired retinal processing layer"""

    def __init__(self, in_channels=1, out_channels=32, adaptive_channels=True):
        super().__init__()

        # Adapt to input channels (1 for MNIST, 3 for CIFAR/ImageNet)
        self.in_channels = in_channels

        # Center-surround receptive fields (like retinal ganglion cells)
        # Different kernel sizes for multi-scale processing
        self.center_on = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.center_off = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.surround_on = nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, stride=1, padding=3)
        self.surround_off = nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, stride=1, padding=3)

        # Combine channels after center-surround
        self.combine = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Lateral inhibition
        self.lateral_inhibition = nn.Conv2d(out_channels, out_channels,
                                            kernel_size=3, padding=1,
                                            groups=out_channels)

        # Adaptive gain control (like retinal adaptation)
        self.gain_control = nn.Parameter(torch.ones(out_channels))

        # Pooling layer to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((14, 14))  # Standard size for vision models

    def forward(self, x):
        # Center-surround processing (ON/OFF pathways)
        center_on = F.relu(self.center_on(x))
        center_off = F.relu(-self.center_off(x))
        surround_on = F.relu(self.surround_on(x))
        surround_off = F.relu(-self.surround_off(x))

        # Combine all pathways
        combined = torch.cat([center_on, center_off, surround_on, surround_off], dim=1)
        combined = self.combine(combined)

        # Lateral inhibition
        inhibited = combined - 0.1 * F.relu(self.lateral_inhibition(combined))

        # Adaptive gain control
        gain = torch.sigmoid(self.gain_control).view(1, -1, 1, 1)
        output = inhibited * gain

        # Adaptive pooling to standard size
        output = self.pool(output)

        return output


class FlexibleCMCRetina(nn.Module):
    """
    Flexible CMC with retinal preprocessing
    Suitable for multiple vision tasks (MNIST, CIFAR-10, ImageNet)
    """

    def __init__(self, num_nodes=8, hidden_dim=128, num_classes=10,
                 in_channels=1, retinal_channels=32, input_size=(28, 28),
                 track_dynamics=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.track_dynamics = track_dynamics
        self.input_size = input_size

        # Retinal processing layer
        self.retina = RetinalConvLayer(in_channels=in_channels,
                                       out_channels=retinal_channels)

        # Calculate feature map size after retinal processing
        # After adaptive pooling: 14x14
        feature_map_size = 14 * 14 * retinal_channels

        # Projection to CMC input
        self.input_projection = nn.Sequential(
            nn.Linear(feature_map_size, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_nodes * 8)  # 8 states per node
        )

        # FlexibleCMC dynamics
        self.cmc = FlexibleCMC(num_nodes=num_nodes, batch_size=32)

        # Output classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * 8, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # For tracking dynamics
        self.trajectories = []

    def forward(self, x, return_trajectory=False, time_span=[0, 1], num_time_points=20):
        batch_size = x.shape[0]

        # Update CMC batch size if needed
        if batch_size != self.cmc.batch_size:
            self.cmc.batch_size = batch_size
            self.cmc.reset_hidden()

        # Retinal processing
        retinal_output = self.retina(x)

        # Flatten spatial dimensions
        retinal_features = retinal_output.flatten(1)

        # Project to CMC initial state
        initial_state = self.input_projection(retinal_features)

        # Time points for ODE solution
        t = torch.linspace(time_span[0], time_span[1], num_time_points).to(x.device)

        # Solve CMC dynamics
        def cmc_dynamics(t, state):
            return self.cmc(t, state)

        try:
            trajectory = odeint(
                cmc_dynamics,
                initial_state,
                t,
                method='euler',
                options={'step_size': 0.05}
            )

            # Track dynamics if requested
            if self.track_dynamics or return_trajectory:
                self.trajectories.append(trajectory.detach().cpu())

            # Use final state for classification
            final_state = trajectory[-1]

        except Exception as e:
            print(f"ODE solver failed: {e}, using initial state")
            final_state = initial_state
            trajectory = None

        output = self.classifier(final_state)

        if return_trajectory:
            return output, trajectory
        return output

    def get_retinal_features(self, x):
        """Extract retinal features for analysis"""
        return self.retina(x)

    def get_cmc_state(self, x):
        """Get the final CMC state without classification"""
        batch_size = x.shape[0]

        if batch_size != self.cmc.batch_size:
            self.cmc.batch_size = batch_size
            self.cmc.reset_hidden()

        retinal_output = self.retina(x)
        retinal_features = retinal_output.flatten(1)
        initial_state = self.input_projection(retinal_features)

        t = torch.linspace(0, 1, 20).to(x.device)

        try:
            trajectory = odeint(
                lambda t, state: self.cmc(t, state),
                initial_state,
                t,
                method='euler',
                options={'step_size': 0.05}
            )
            final_state = trajectory[-1]
        except:
            final_state = initial_state

        return final_state

    def get_phase_space_data(self, dataloader, num_samples=1000):
        """Extract phase space trajectories for different classes"""
        self.eval()
        self.reset_trajectories()

        phase_data = {}
        labels_collected = {}

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                device = next(self.parameters()).device
                data = data.to(device)

                _, trajectories = self.forward(data, return_trajectory=True)

                if trajectories is not None:
                    for i, label in enumerate(target):
                        label = label.item()
                        if label not in phase_data:
                            phase_data[label] = []
                            labels_collected[label] = 0

                        if labels_collected[label] < num_samples // len(set(target.numpy())):
                            phase_data[label].append(
                                trajectories[:, i, :].cpu().numpy()
                            )
                            labels_collected[label] += 1

                # Check if we've collected enough samples
                if all(count >= num_samples // len(phase_data)
                       for count in labels_collected.values()):
                    break

        return phase_data

    def reset_trajectories(self):
        """Clear stored trajectories"""
        self.trajectories = []

    def adapt_to_dataset(self, dataset_name):
        """Adapt model configuration for different datasets"""
        configs = {
            'mnist': {'in_channels': 1, 'num_classes': 10, 'input_size': (28, 28)},
            'cifar10': {'in_channels': 3, 'num_classes': 10, 'input_size': (32, 32)},
            'cifar100': {'in_channels': 3, 'num_classes': 100, 'input_size': (32, 32)},
            'imagenet': {'in_channels': 3, 'num_classes': 1000, 'input_size': (224, 224)},
            'tiny_imagenet': {'in_channels': 3, 'num_classes': 200, 'input_size': (64, 64)}
        }

        if dataset_name.lower() in configs:
            config = configs[dataset_name.lower()]
            print(f"Adapting model for {dataset_name}: {config}")
            # Note: This would require rebuilding the model with new parameters
            return config
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


# Factory function for creating models for specific datasets
def create_retinal_cmc_model(dataset_name, num_nodes=8, hidden_dim=128):
    """Create a FlexibleCMCRetina model configured for a specific dataset"""

    configs = {
        'mnist': {'in_channels': 1, 'num_classes': 10, 'input_size': (28, 28)},
        'cifar10': {'in_channels': 3, 'num_classes': 10, 'input_size': (32, 32)},
        'cifar100': {'in_channels': 3, 'num_classes': 100, 'input_size': (32, 32)},
        'imagenet': {'in_channels': 3, 'num_classes': 1000, 'input_size': (224, 224)},
        'tiny_imagenet': {'in_channels': 3, 'num_classes': 200, 'input_size': (64, 64)}
    }

    if dataset_name.lower() not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = configs[dataset_name.lower()]

    model = FlexibleCMCRetina(
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
        input_size=config['input_size']
    )

    return model