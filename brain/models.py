from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class BrainNetwork(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    is_active = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_accessed = models.DateTimeField(default=timezone.now)

    # STAG Parameters
    sdr_dimensionality = models.IntegerField(default=2048)
    sdr_sparsity = models.IntegerField(default=40)

    # GNG Parameters
    max_nodes = models.IntegerField(default=5000)
    n_iter_before_neuron_added = models.IntegerField(default=50) # Reduced for faster growth
    max_edge_age = models.IntegerField(default=50)
    winner_learning_rate = models.FloatField(default=0.08)
    neighbor_learning_rate = models.FloatField(default=0.005)
    error_decay_rate = models.FloatField(default=0.9995)

    # HTM Parameters
    cells_per_column = models.IntegerField(default=16)
    activation_threshold = models.IntegerField(default=13)
    initial_permanence = models.FloatField(default=0.21)
    connected_permanence = models.FloatField(default=0.5)
    permanence_increment = models.FloatField(default=0.1)
    permanence_decrement = models.FloatField(default=0.05)

    # RL Parameters
    rl_learning_rate = models.FloatField(default=0.05)
    rl_discount_factor = models.FloatField(default=0.95)
    rl_exploration_rate = models.FloatField(default=1.0)

    def __str__(self):
        return f"{self.name} (ID: {self.id})"

    class Meta:
        ordering = ['-created_at']
        verbose_name = "STAG Brain Network"


class GraphSnapshot(models.Model):
    network = models.ForeignKey(BrainNetwork, related_name='snapshots', on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, help_text="e.g., 'After training on Maze env for 100 episodes'")

    class Meta:
        unique_together = ('network', 'name')
        ordering = ['-created_at']

    def __str__(self):
        return f"Snapshot '{self.name}' for {self.network.name}"


class SnapshotNeuron(models.Model):
    snapshot = models.ForeignKey(GraphSnapshot, related_name='neurons', on_delete=models.CASCADE)
    neuron_id = models.IntegerField()
    prototype_sdr = models.JSONField()
    error = models.FloatField(default=0.0)
    cells = models.JSONField()
    last_active_iter = models.IntegerField(default=0)
    node_type = models.CharField(max_length=50, default='sensory')
    action_name = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=['snapshot', 'neuron_id'])]


class SnapshotEdge(models.Model):
    snapshot = models.ForeignKey(GraphSnapshot, related_name='edges', on_delete=models.CASCADE)
    source_id = models.IntegerField()
    target_id = models.IntegerField()
    age = models.IntegerField(default=0)

    class Meta:
        indexes = [models.Index(fields=['snapshot'])]


class TrainingSession(models.Model):
    """Records a single training run for a network."""
    class Status(models.TextChoices):
        RUNNING = 'RUNNING', 'Running'
        COMPLETED = 'COMPLETED', 'Completed'
        FAILED = 'FAILED', 'Failed'

    network = models.ForeignKey(BrainNetwork, related_name='training_sessions', on_delete=models.CASCADE)
    environment_name = models.CharField(max_length=100)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.RUNNING)
    parameters = models.JSONField(help_text="Parameters like episodes, max_steps, etc.")
    results = models.JSONField(null=True, blank=True, help_text="Final results, e.g., rewards per episode")

    class Meta:
        ordering = ['-start_time']

    def __str__(self):
        return f"Training on {self.environment_name} for {self.network.name} at {self.start_time}"
