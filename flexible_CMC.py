import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexibleCMC(nn.Module):
    def __init__(self, num_nodes=4, batch_size=32, activation='sigmoid'):
        super().__init__()
        self.num_nodes = num_nodes
        self.batch_size = batch_size

        # Learnable connectivity matrix for inter-node connections
        self.node_connectivity = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)

        # Per-node parameters (preserved from original)
        self.G = nn.ParameterList([
            nn.Parameter(torch.tensor([8., 8., 16., 8., 8., 4., 8., 8., 4., 4., 4., 8.]))
            for _ in range(num_nodes)
        ])
        self.T = nn.ParameterList([
            nn.Parameter(torch.tensor([0.02, 0.02, 0.16, 0.28]))
            for _ in range(num_nodes)
        ])

        # Learnable adaptation parameters
        self.adaptation = nn.Parameter(torch.ones(num_nodes))
        self.R = nn.Parameter(torch.ones(num_nodes))

        self.activation = activation
        self.reset_hidden()

    def reset_hidden(self):
        # Hidden state for each node
        self.hidden = torch.zeros(self.batch_size, self.num_nodes * 8)

    def activation_function(self, x, node_idx):
        if self.activation == 'sigmoid':
            return torch.sigmoid(self.R[node_idx] * x)
        elif self.activation == 'softplus':
            return F.softplus(self.R[node_idx] * x)
        elif self.activation == 'tanh':
            return torch.tanh(self.R[node_idx] * x)
        return F.relu(x)

    def forward(self, t, x):
        # Ensure proper batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = torch.clamp(x, -100, 100)
        f = torch.zeros_like(x)

        for i in range(self.num_nodes):
            start_idx = i * 8
            end_idx = start_idx + 8

            # Get current node's state
            node_x = x[:, start_idx:end_idx]
            node_f = torch.zeros_like(node_x)

            # Calculate firing rates
            F = self.activation_function(node_x, i)
            if self.activation == 'sigmoid':
                S = F - self.activation_function(torch.zeros_like(node_x), i)
            else:
                S = F

            # Inter-node influence (FIXED: ensure node_input is properly shaped)
            node_input = torch.zeros(x.shape[0], 1, device=x.device)
            for j in range(self.num_nodes):
                if i != j:
                    other_start = j * 8
                    other_activity = x[:, other_start:other_start + 8].mean(dim=1, keepdim=True)
                    node_input += self.node_connectivity[i, j] * other_activity

            # Expand node_input to match the shape needed for the equations
            node_input_expanded = node_input.expand(-1, node_x.shape[1])

            # Conductance equations with adaptive scaling
            scale = torch.sigmoid(self.adaptation[i]) * 0.1

            # Update equations - preserving all original dynamics
            # FIXED: Use expanded node_input for proper broadcasting
            u = -self.G[i][0] * S[:, 0] - self.G[i][2] * S[:, 4] - self.G[i][1] * S[:, 2] + node_input_expanded[:, 0]
            node_f[:, 1] = scale * (u - 2 * node_x[:, 1] - node_x[:, 0] / self.T[i][0]) / self.T[i][0]

            u = self.G[i][7] * S[:, 0] - self.G[i][6] * S[:, 2] - self.G[i][11] * S[:, 4] + node_input_expanded[:, 0]
            node_f[:, 3] = scale * (u - 2 * node_x[:, 3] - node_x[:, 2] / self.T[i][1]) / self.T[i][1]

            u = (self.G[i][4] * S[:, 0] + self.G[i][5] * S[:, 6] - self.G[i][3] * S[:, 4] +
                 self.G[i][10] * S[:, 2] + node_input_expanded[:, 0])
            node_f[:, 5] = scale * (u - 2 * node_x[:, 5] - node_x[:, 4] / self.T[i][2]) / self.T[i][2]

            u = -self.G[i][9] * S[:, 6] - self.G[i][8] * S[:, 4] + node_input_expanded[:, 0]
            node_f[:, 7] = scale * (u - 2 * node_x[:, 7] - node_x[:, 6] / self.T[i][3]) / self.T[i][3]

            # Voltage equations (preserved exactly as in original)
            node_f[:, 0] = scale * node_x[:, 1]
            node_f[:, 2] = scale * node_x[:, 3]
            node_f[:, 4] = scale * node_x[:, 5]
            node_f[:, 6] = scale * node_x[:, 7]

            f[:, start_idx:end_idx] = node_f

        return f

    def get_population_states(self, x):
        """
        Extract states for each cell population
        Useful for visualization of dynamics

        Returns:
            dict: States for each population in each node
        """
        population_states = {}

        for i in range(self.num_nodes):
            start_idx = i * 8
            node_states = x[:, start_idx:start_idx + 8]

            # Each node has 4 populations with 2 states each (voltage and current)
            population_states[f'node_{i}'] = {
                'pyramidal_voltage': node_states[:, 0],
                'pyramidal_current': node_states[:, 1],
                'stellate_voltage': node_states[:, 2],
                'stellate_current': node_states[:, 3],
                'inhibitory_voltage': node_states[:, 4],
                'inhibitory_current': node_states[:, 5],
                'deep_pyramidal_voltage': node_states[:, 6],
                'deep_pyramidal_current': node_states[:, 7]
            }

        return population_states