from django.db import models
from django.core.exceptions import ValidationError

class BrainNetwork(models.Model):
    """
    Represents a single STAG network instance and its hyperparameters.
    """
    name = models.CharField(max_length=255, unique=True)

    # --- STAG Hyperparameters ---
    sdr_dimensionality = models.PositiveIntegerField(default=2048, help_text="Total number of bits in an SDR.")
    sdr_sparsity = models.PositiveIntegerField(default=40, help_text="Number of active bits in an SDR (approx 2%).")

    # GNG Learning rates and decay
    winner_learning_rate = models.FloatField(default=0.1, help_text="Learning rate (eta_1) for the winning node.")
    neighbor_learning_rate = models.FloatField(default=0.01,
                                               help_text="Learning rate (eta_2) for the winner's neighbors.")
    error_decay_rate = models.FloatField(default=0.9995, help_text="Decay factor for all nodes' errors per iteration.")

    # GNG Structural plasticity params
    max_edge_age = models.PositiveIntegerField(default=50, help_text="Max age for a topological edge before it's pruned.")
    n_iter_before_neuron_added = models.PositiveIntegerField(default=100,
                                                             help_text="Number of iterations before growing a new node.")
    max_nodes = models.PositiveIntegerField(default=5000, help_text="The maximum number of nodes the network can grow to.")

    # HTM Temporal Sequence Hyperparameters
    cells_per_column = models.PositiveIntegerField(default=32, help_text="Number of cells within each node (mini-column).")
    initial_permanence = models.FloatField(default=0.21, help_text="Initial permanence for new synapses.")
    connected_permanence = models.FloatField(default=0.50, help_text="Permanence value above which a synapse is considered connected.")
    permanence_increment = models.FloatField(default=0.10, help_text="Amount to increase permanence for active synapses.")
    permanence_decrement = models.FloatField(default=0.02, help_text="Amount to decrease permanence for inactive synapses.")
    activation_threshold = models.PositiveIntegerField(default=10, help_text="Number of connected synapses required to activate a dendrite segment.")

    # --- Reinforcement Learning Hyperparameters ---
    rl_learning_rate = models.FloatField(default=0.1, help_text="Q-learning learning rate (alpha).")
    rl_discount_factor = models.FloatField(default=0.9, help_text="RL discount factor (gamma) for future rewards.")
    rl_exploration_rate = models.FloatField(default=0.3, help_text="RL exploration rate (epsilon) for choosing random actions.")

    is_active = models.BooleanField(default=False, help_text="Designates this as the currently active network. Only one can be active.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Brain Network"
        verbose_name_plural = "Brain Networks"

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        """
        Ensures that only one network can be active at a time.
        """
        if self.is_active:
            BrainNetwork.objects.filter(is_active=True).exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)

class GraphSnapshot(models.Model):
    """
    Represents a saved state (snapshot) of a STAG network at a point in time.
    This allows for versioning and restoring the graph's "weights".
    """
    network = models.ForeignKey(BrainNetwork, related_name='snapshots', on_delete=models.CASCADE)
    name = models.CharField(max_length=255, help_text="A descriptive name for the snapshot.")
    graph_data = models.JSONField(help_text="The entire graph state (nodes and links) serialized as JSON.")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Graph Snapshot"
        verbose_name_plural = "Graph Snapshots"
        unique_together = ('network', 'name')

    def __str__(self):
        return f"Snapshot '{self.name}' for {self.network.name} @ {self.created_at.strftime('%Y-%m-%d %H:%M')}"
