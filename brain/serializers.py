from rest_framework import serializers
from .models import BrainNetwork, GraphSnapshot, TrainingSession, SnapshotNeuron, SnapshotEdge

class BrainNetworkSerializer(serializers.ModelSerializer):
    """Basic serializer for BrainNetwork, used for lists."""
    class Meta:
        model = BrainNetwork
        fields = [
            'id', 'name', 'description', 'is_active',
            'created_at', 'updated_at', 'last_accessed'
        ]

class TrainingSessionSerializer(serializers.ModelSerializer):
    """Serializer for TrainingSession records."""
    class Meta:
        model = TrainingSession
        fields = '__all__'

class BrainNetworkDetailSerializer(serializers.ModelSerializer):
    """
    Detailed serializer for a single BrainNetwork instance, including
    its parameters and recent training history.
    """
    training_sessions = TrainingSessionSerializer(many=True, read_only=True)

    class Meta:
        model = BrainNetwork
        fields = '__all__' # Includes all fields from the model plus the related sessions

class GraphSnapshotSerializer(serializers.ModelSerializer):
    """Serializer for GraphSnapshot, used for lists and creation."""
    class Meta:
        model = GraphSnapshot
        fields = ['id', 'network', 'name', 'created_at', 'notes']

# --- Fix for ImportError ---

class SnapshotNeuronSerializer(serializers.ModelSerializer):
    """Serializer for individual neuron data within a snapshot."""
    class Meta:
        model = SnapshotNeuron
        # Exclude the foreign key to the snapshot itself to avoid redundancy
        exclude = ['snapshot']

class SnapshotEdgeSerializer(serializers.ModelSerializer):
    """Serializer for individual edge data within a snapshot."""
    class Meta:
        model = SnapshotEdge
        # Exclude the foreign key to the snapshot itself
        exclude = ['snapshot']

class GraphSnapshotDetailSerializer(serializers.ModelSerializer):
    """
    Detailed serializer for a single GraphSnapshot, including all its
    neurons and edges for reconstruction or detailed viewing.
    """
    # Nested serializers for related neurons and edges
    neurons = SnapshotNeuronSerializer(many=True, read_only=True)
    edges = SnapshotEdgeSerializer(many=True, read_only=True)

    class Meta:
        model = GraphSnapshot
        fields = ['id', 'network', 'name', 'created_at', 'notes', 'neurons', 'edges']
