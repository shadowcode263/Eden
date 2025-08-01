from django.db import models
import numpy as np
import pickle
import hashlib
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BrainNetwork(models.Model):
    name = models.CharField(max_length=255)
    embedding_dim = models.IntegerField(default=64)
    beta = models.FloatField(default=20.0)
    learning_rate = models.FloatField(default=0.1)
    merkle_root = models.CharField(max_length=64, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'brain_networks'

    def __str__(self):
        return f"Brain Network: {self.name}"


class BrainPattern(models.Model):
    network = models.ForeignKey(BrainNetwork, on_delete=models.CASCADE, related_name='patterns')
    pattern_hash = models.CharField(max_length=64, unique=True)
    text_content = models.TextField(null=True, blank=True) # New field to store original text
    pattern_data = models.BinaryField()  # Serialized numpy array
    embedding_data = models.BinaryField()  # Serialized embedding
    usage_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'brain_patterns'
        # Add a unique_together constraint to ensure pattern_hash is unique per network
        # This prevents issues if the same hash could exist across different networks
        unique_together = ('network', 'pattern_hash',)


    def get_pattern_array(self) -> np.ndarray:
        try:
            return pickle.loads(self.pattern_data)
        except Exception as e:
            logger.error(f"Error deserializing pattern_data for hash {self.pattern_hash}: {e}")
            # Return an empty array or raise a specific error if data is corrupted
            return np.array([])

    def set_pattern_array(self, pattern: np.ndarray):
        try:
            self.pattern_data = pickle.dumps(pattern)
            logger.debug(f"Pattern {self.pattern_hash[:8]}... pickled. Size: {len(self.pattern_data)} bytes.")
        except Exception as e:
            logger.error(f"Error pickling pattern_data for hash {self.pattern_hash}: {e}")
            raise # Re-raise to indicate a critical error

    def get_embedding_array(self) -> np.ndarray:
        try:
            return pickle.loads(self.embedding_data)
        except Exception as e:
            logger.error(f"Error deserializing embedding_data for hash {self.pattern_hash}: {e}")
            # Return an empty array or raise a specific error if data is corrupted
            return np.array([])

    def set_embedding_array(self, embedding: np.ndarray):
        try:
            self.embedding_data = pickle.dumps(embedding)
            logger.debug(f"Embedding {self.pattern_hash[:8]}... pickled. Size: {len(self.embedding_data)} bytes.")
        except Exception as e:
            logger.error(f"Error pickling embedding_data for hash {self.pattern_hash}: {e}")
            raise # Re-raise to indicate a critical error

    def __str__(self):
        return f"Pattern {self.pattern_hash[:8]}..."


class BrainRetrieval(models.Model):
    network = models.ForeignKey(BrainNetwork, on_delete=models.CASCADE, related_name='retrievals')
    query_text = models.TextField(null=True, blank=True)
    retrieved_pattern = models.ForeignKey(BrainPattern, on_delete=models.SET_NULL, null=True)
    confidence_score = models.FloatField()
    # Changed from IntegerField to JSONField to store the full steps data
    retrieval_steps_data = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'brain_retrievals'

    def __str__(self):
        return f"Retrieval at {self.created_at}"
