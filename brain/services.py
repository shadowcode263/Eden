import numpy as np
import hashlib
import pickle
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from sklearn.decomposition import PCA
from django.core.cache import cache
import redis
from django.conf import settings
from .models import BrainNetwork, BrainPattern, BrainRetrieval
import logging

logger = logging.getLogger(__name__)


class DynamicMerkleTree:
    def __init__(self):
        self.leaves = []
        self.tree = []

    def _hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    def add_leaf(self, data: str):
        self.leaves.append(data)

    def build_tree(self):
        if not self.leaves:
            self.tree = []
            return

        level = [self._hash(leaf) for leaf in self.leaves]
        self.tree = [level]

        while len(level) > 1:
            next_level = []
            if len(level) % 2 == 1:
                level.append(level[-1])

            for i in range(0, len(level), 2):
                combined_hash = self._hash(level[i] + level[i + 1])
                next_level.append(combined_hash)

            self.tree.append(next_level)
            level = next_level

    def get_root(self) -> str:
        return self.tree[-1][0] if self.tree and self.tree[-1] else None


class ModernHopfieldLayer:
    def __init__(self, beta: float = 10.0):
        self.beta = beta

    def __call__(self, stored_patterns: np.ndarray, query_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scores = stored_patterns.T @ query_vector
        scaled_scores = self.beta * (scores - np.max(scores))
        exp_scores = np.exp(scaled_scores)
        weights = exp_scores / (np.sum(exp_scores) + 1e-9)
        new_state = stored_patterns @ weights
        return new_state, weights


class HopfieldNetworkService:
    def __init__(self, network_id: int):
        self.network = BrainNetwork.objects.get(id=network_id)
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.modern_hopfield_layer = ModernHopfieldLayer(beta=self.network.beta)
        self.merkle = DynamicMerkleTree()
        self._load_network_state()

    def _load_network_state(self):
        """Load network state from database and cache"""
        cache_key = f"network_{self.network.id}_state"
        cached_state = cache.get(cache_key)

        if cached_state:
            logger.info(f"Loaded network {self.network.id} state from cache.")
            self.stored_matrix = cached_state['stored_matrix']
            self.patterns = cached_state['patterns']
            self.embeddings = cached_state['embeddings']
        else:
            logger.info(f"Cache miss for network {self.network.id} state. Rebuilding from database.")
            self._rebuild_from_database()
            self._cache_network_state()

    def _rebuild_from_database(self):
        """Rebuild network state from database"""
        patterns = self.network.patterns.all()
        self.patterns = {}
        self.embeddings = {}
        self.merkle = DynamicMerkleTree()  # Re-initialize Merkle tree

        if patterns.exists():
            embedding_list = []
            for pattern in patterns:
                try:
                    pattern_array = pattern.get_pattern_array()
                    embedding_array = pattern.get_embedding_array()

                    self.patterns[pattern.pattern_hash] = {
                        'pattern_array': pattern_array,
                        'text_content': pattern.text_content  # Store text content in memory
                    }
                    self.embeddings[pattern.pattern_hash] = embedding_array
                    embedding_list.append(embedding_array)
                    self.merkle.add_leaf(pattern.pattern_hash)
                except Exception as e:
                    logger.error(f"Error loading pattern {pattern.pattern_hash} from DB: {e}")
                    continue  # Skip corrupted pattern

            if embedding_list:
                self.stored_matrix = np.array(embedding_list).T
                self.merkle.build_tree()
                logger.info(
                    f"Rebuilt network {self.network.id} with {len(self.patterns)} patterns. Merkle Root: {self.merkle.get_root()}")
            else:
                self.stored_matrix = None
                logger.info(f"No valid patterns found for network {self.network.id}.")
        else:
            self.stored_matrix = None
            logger.info(f"No patterns in database for network {self.network.id}.")

    def _cache_network_state(self):
        """Cache network state in Redis"""
        cache_key = f"network_{self.network.id}_state"
        state = {
            'stored_matrix': self.stored_matrix,
            'patterns': self.patterns,
            'embeddings': self.embeddings
        }
        try:
            cache.set(cache_key, state, timeout=3600)  # Cache for 1 hour
            logger.info(f"Network {self.network.id} state cached successfully.")
        except Exception as e:
            logger.error(f"Error caching network state for {self.network.id}: {e}")

    def _initialize_embedding(self, pattern: np.ndarray) -> np.ndarray:
        embedding = pattern.astype(np.float32)
        logger.debug(f"Initial embedding shape: {embedding.shape}, dtype: {embedding.dtype}")

        if len(embedding) > self.network.embedding_dim:
            pca = PCA(n_components=self.network.embedding_dim, random_state=42)
            embedding = pca.fit_transform(embedding.reshape(1, -1)).flatten()
            logger.debug(f"PCA applied. New embedding shape: {embedding.shape}")
        elif len(embedding) < self.network.embedding_dim:
            padding = np.zeros(self.network.embedding_dim - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
            # logger.debug(f"Padding applied. New embedding shape: {normalized_embedding.shape}")

        norm = np.linalg.norm(embedding)
        normalized_embedding = embedding / norm if norm > 0 else embedding
        logger.debug(f"Normalized embedding shape: {normalized_embedding.shape}, norm: {norm}")
        return normalized_embedding

    def vectorize_string(self, input_str: str, length: int) -> np.ndarray:
        """Convert string to vector representation"""
        padded_str = input_str.ljust(length, '\x00')[:length]
        ascii_vals = np.array([ord(c) for c in padded_str], dtype=np.float32)
        norm = np.linalg.norm(ascii_vals)
        vector = ascii_vals / norm if norm > 0 else ascii_vals
        logger.debug(f"Vectorized string '{input_str[:10]}...'. Shape: {vector.shape}, dtype: {vector.dtype}")
        return vector

    def store_pattern(self, text: str) -> Dict[str, Any]:
        """Store a new pattern in the network"""
        # Convert text to pattern
        max_len = max(len(text), 32)  # Minimum length
        pattern = self.vectorize_string(text, max_len)
        logger.debug(
            f"Pattern generated from text. Type: {type(pattern)}, Shape: {pattern.shape}, Dtype: {pattern.dtype}")

        # Create hash
        pattern_hash = hashlib.sha256(pattern.tobytes()).hexdigest()
        logger.debug(f"Pattern hash: {pattern_hash}")

        # Check if already exists
        if BrainPattern.objects.filter(network=self.network, pattern_hash=pattern_hash).exists():
            logger.info(f"Pattern with hash {pattern_hash} already exists for network {self.network.id}.")
            return {'success': False, 'message': 'Pattern already exists'}

        # Create embedding
        embedding = self._initialize_embedding(pattern)
        logger.debug(
            f"Embedding generated. Type: {type(embedding)}, Shape: {embedding.shape}, Dtype: {embedding.dtype}")

        # Save to database
        try:
            brain_pattern = BrainPattern.objects.create(
                network=self.network,
                pattern_hash=pattern_hash,
                text_content=text,  # Store the original text content
                usage_count=0
            )
            brain_pattern.set_pattern_array(pattern)
            brain_pattern.set_embedding_array(embedding)
            brain_pattern.save()
            logger.info(f"Pattern {pattern_hash} saved to database.")
        except Exception as e:
            logger.error(f"Error saving pattern to database: {e}")
            return {'success': False, 'message': f'Failed to save pattern: {e}'}

        # Update in-memory state
        self.patterns[pattern_hash] = {
            'pattern_array': pattern,
            'text_content': text
        }
        self.embeddings[pattern_hash] = embedding

        if self.stored_matrix is None:
            self.stored_matrix = embedding.reshape(-1, 1)
        else:
            # Ensure dimensions match before hstack
            if self.stored_matrix.shape[0] != embedding.shape[0]:
                logger.warning(
                    f"Dimension mismatch: stored_matrix {self.stored_matrix.shape[0]} vs embedding {embedding.shape[0]}. Reshaping embedding.")
                # This might indicate an issue with embedding_dim consistency or PCA
                # For now, we'll try to reshape, but it's a sign to investigate
                embedding_reshaped = embedding.reshape(-1, 1)
                if embedding_reshaped.shape[0] != self.stored_matrix.shape[0]:
                    logger.error("Cannot stack embeddings due to incompatible dimensions.")
                    return {'success': False, 'message': 'Incompatible embedding dimensions'}
                self.stored_matrix = np.hstack([self.stored_matrix, embedding_reshaped])
            else:
                self.stored_matrix = np.hstack([self.stored_matrix, embedding.reshape(-1, 1)])
        logger.debug(f"Stored matrix updated. New shape: {self.stored_matrix.shape}")

        # Update Merkle tree
        self.merkle.add_leaf(pattern_hash)
        self.merkle.build_tree()
        logger.debug(f"Merkle tree rebuilt. Root: {self.merkle.get_root()}")

        # Update network's merkle root
        self.network.merkle_root = self.merkle.get_root()
        self.network.save()
        logger.info(f"Network {self.network.id} Merkle root updated to {self.network.merkle_root}.")

        # Update cache
        self._cache_network_state()

        # Publish to WebSocket
        self._publish_network_update('pattern_stored', {
            'pattern_hash': pattern_hash,
            'text': text,
            'total_patterns': len(self.patterns),
            'merkle_root': self.network.merkle_root  # Include Merkle root in update
        })

        return {
            'success': True,
            'pattern_hash': pattern_hash,
            'total_patterns': len(self.patterns),
            'merkle_root': self.network.merkle_root
        }

    def retrieve_pattern(self, query_text: str, max_iter: int = 10) -> Dict[str, Any]:
        """Retrieve a pattern from the network"""
        if self.stored_matrix is None or self.stored_matrix.size == 0:
            logger.warning(f"Attempted retrieval on network {self.network.id} with no patterns stored.")
            return {'success': False, 'message': 'No patterns stored'}

        # Convert query to pattern
        max_len = max(len(query_text), 32)
        query_pattern = self.vectorize_string(query_text, max_len)
        query_embedding = self._initialize_embedding(query_pattern)
        logger.debug(f"Query embedding generated. Shape: {query_embedding.shape}")

        # Ensure query embedding dimension matches stored matrix dimension
        if query_embedding.shape[0] != self.stored_matrix.shape[0]:
            logger.error(
                f"Query embedding dimension mismatch: {query_embedding.shape[0]} vs stored matrix {self.stored_matrix.shape[0]}. Cannot retrieve.")
            return {'success': False, 'message': 'Query embedding dimension mismatch with stored patterns.'}

        # Perform retrieval
        current_state = query_embedding.copy()
        retrieval_steps = []

        for step in range(max_iter):
            new_state, weights = self.modern_hopfield_layer(self.stored_matrix, current_state)

            # Store step for visualization
            retrieval_steps.append({
                'step': step,
                'state': current_state.tolist(),
                'weights': weights.tolist()
            })
            logger.debug(
                f"Retrieval step {step}: state (first 5) = {current_state[:5].tolist()}, weights (first 5) = {weights[:5].tolist()}")

            if np.linalg.norm(new_state - current_state) < 1e-5:
                logger.debug(f"Retrieval converged in {step + 1} steps.")
                break

            current_state = new_state
        else:
            logger.debug(f"Retrieval reached max iterations ({max_iter}) without full convergence.")

        logger.debug(f"Total retrieval steps generated: {len(retrieval_steps)}")
        if retrieval_steps:
            logger.debug(f"Last step weights (first 5): {retrieval_steps[-1]['weights'][:5]}")

        # Find best match
        if weights.size == 0:
            logger.warning("No weights generated during retrieval, likely no patterns.")
            return {'success': False, 'message': 'No patterns to retrieve from.'}

        best_index = np.argmax(weights)
        best_hash = list(self.patterns.keys())[best_index]
        confidence_score = float(weights[best_index])
        logger.info(f"Retrieved pattern {best_hash[:8]}... with confidence {confidence_score:.2f}.")

        # Get the retrieved pattern object from DB to access its text_content
        try:
            pattern = BrainPattern.objects.get(pattern_hash=best_hash, network=self.network)
            retrieved_text_content = pattern.text_content
            pattern.usage_count += 1
            pattern.save()
            logger.debug(f"Usage count for pattern {best_hash[:8]}... incremented.")
        except BrainPattern.DoesNotExist:
            logger.error(f"Retrieved pattern {best_hash} not found in DB for usage count update.")
            retrieved_text_content = None  # Fallback if pattern not found

        # Save retrieval record
        retrieval = BrainRetrieval.objects.create(
            network=self.network,
            query_text=query_text,
            retrieved_pattern=pattern,
            confidence_score=confidence_score,
            retrieval_steps_data=retrieval_steps  # Store the full steps data
        )
        logger.info(f"Retrieval record created: {retrieval.id}")

        # Publish to WebSocket
        self._publish_network_update('pattern_retrieved', {
            'query_text': query_text,
            'retrieved_hash': best_hash,
            'retrieved_text': retrieved_text_content,  # Include retrieved text
            'confidence': confidence_score,
            'steps': retrieval_steps,
            'merkle_root': self.network.merkle_root  # Include Merkle root in update
        })

        return {
            'success': True,
            'retrieved_hash': best_hash,
            'retrieved_text': retrieved_text_content,  # Include retrieved text
            'confidence_score': confidence_score,
            'retrieval_steps': retrieval_steps,
            'pattern_data': self.patterns[best_hash]['pattern_array'].tolist()  # Access pattern_array from dict
        }

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        patterns = self.network.patterns.all()
        retrievals = self.network.retrievals.all()

        # Ensure Merkle tree is up-to-date before getting root
        if not self.merkle.tree or self.merkle.leaves != [p.pattern_hash for p in patterns.order_by('created_at')]:
            logger.info("Merkle tree out of sync, rebuilding for stats.")
            self._rebuild_from_database()  # Rebuild to ensure current state

        return {
            'total_patterns': patterns.count(),
            'total_retrievals': retrievals.count(),
            'embedding_dim': self.network.embedding_dim,
            'merkle_root': self.merkle.get_root(),
            'most_used_patterns': [
                {
                    'hash': p.pattern_hash,
                    'usage_count': p.usage_count
                }
                for p in patterns.order_by('-usage_count')[:5]
            ]
        }

    def _publish_network_update(self, event_type: str, data: Dict[str, Any]):
        """Publish network updates to WebSocket"""
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync

        channel_layer = get_channel_layer()
        try:
            async_to_sync(channel_layer.group_send)(
                f"network_{self.network.id}",
                {
                    'type': 'network_update',
                    'event_type': event_type,
                    'data': data
                }
            )
            logger.debug(f"Published WebSocket update: {event_type} for network {self.network.id}")
        except Exception as e:
            logger.error(f"Failed to publish WebSocket update for network {self.network.id}: {e}")
