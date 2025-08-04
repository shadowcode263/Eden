import random
import math
import time
from typing import List, Set, Tuple, Dict, Any, Generator, Union
from collections import Counter, deque, namedtuple

import networkx as nx
import numpy as np
from tqdm import tqdm
from django.db import transaction
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from pybloom_live import ScalableBloomFilter

try:
    import faiss
except ImportError:
    faiss = None
    print("Warning: FAISS library not found. Falling back to slower similarity search.")

try:
    import jax
    import jax.numpy as jnp
    from jax import random as jax_random

    JAX_AVAILABLE = True
    print("JAX library found. Using JAX for accelerated computations.")
except ImportError:
    jnp = np  # Fallback to numpy if JAX is not available
    JAX_AVAILABLE = False
    print("Warning: JAX library not found. Some computations may be slower.")

from .models import BrainNetwork, GraphSnapshot, SnapshotNeuron, SnapshotEdge
from .serializers import GraphSnapshotDetailSerializer
from .environments import BaseEnvironment

# --- Constants ---
SNAPSHOT_SCHEMA_VERSION = "2.7"
CELL_ID_MULTIPLIER = 10000

# --- Data Structures for Experience Replay ---
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


# --- Core Service Logic ---

class SDRHelper:
    """Utility class for handling Sparse Distributed Representations (SDRs)."""

    def __init__(self, dimensionality: int, sparsity: int):
        if sparsity > dimensionality:
            raise ValueError("Sparsity cannot be greater than dimensionality.")
        self.dimensionality = dimensionality
        self.sparsity = sparsity
        self.encoder_cache: Dict[str, Set[int]] = {}
        self.bloom = ScalableBloomFilter(error_rate=0.001)

    def encode_token(self, token: str) -> Set[int]:
        if token not in self.encoder_cache:
            sdr = self._create_random_sdr()
            self.encoder_cache[token] = sdr
            for bit in sdr:
                self.bloom.add(str(bit))
        return self.encoder_cache[token]

    def _create_random_sdr(self) -> Set[int]:
        return set(random.sample(range(self.dimensionality), self.sparsity))

    @staticmethod
    def union(sdr_list: List[Set[int]]) -> Set[int]:
        return set.union(*sdr_list) if sdr_list else set()

    @staticmethod
    def overlap(sdr1: Set[int], sdr2: Set[int]) -> int:
        return len(sdr1.intersection(sdr2))

    @staticmethod
    def sdr_to_list(sdr: Set[int]) -> List[int]:
        return sorted(list(sdr))

    @staticmethod
    def list_to_sdr(sdr_list: List[int]) -> Set[int]:
        return set(sdr_list)


class StabilityTracker:
    def __init__(self, network_model: BrainNetwork):
        self.base_gng_lr = network_model.winner_learning_rate
        self.base_rl_lr = network_model.rl_learning_rate
        self.growth_rate_window = deque(maxlen=100)
        self.q_volatility_window = deque(maxlen=100)
        self.max_growth_rate = 0.05
        self.max_q_volatility = 0.1

    def track_graph_growth(self, nodes_added: int):
        self.growth_rate_window.append(nodes_added)

    def track_q_volatility(self, q_diff: float):
        self.q_volatility_window.append(q_diff)

    def get_adaptive_gng_lr(self) -> float:
        avg_growth = sum(self.growth_rate_window) / (len(self.growth_rate_window) or 1)
        factor = 1.0 - min(1.0, avg_growth / self.max_growth_rate)
        return self.base_gng_lr * (0.1 + 0.9 * factor)

    def get_adaptive_rl_lr(self) -> float:
        avg_volatility = sum(self.q_volatility_window) / (len(self.q_volatility_window) or 1)
        factor = 1.0 - min(1.0, avg_volatility / self.max_q_volatility)
        return self.base_rl_lr * (0.1 + 0.9 * factor)


class HTMLearner:
    def __init__(self, graph: nx.Graph, network_model: BrainNetwork):
        self.graph = graph
        self.network_model = network_model
        self.dendrite_index: Dict[int, List[Tuple[int, int, int]]] = self._build_dendrite_index()

    def _build_dendrite_index(self) -> Dict[int, List[Tuple[int, int, int]]]:
        index = {}
        for node_id, node_data in self.graph.nodes(data=True):
            cells_dict = node_data.get('cells', {})
            if not isinstance(cells_dict, dict): continue
            for cell_id_str, cell_data in cells_dict.items():
                try:
                    cell_id_int = int(cell_id_str)
                except ValueError:
                    continue

                if not isinstance(cell_data, dict): continue
                for seg_idx, segment in enumerate(cell_data.get('dendrites', [])):
                    if not isinstance(segment, dict): continue
                    for syn_cell_id_str in segment.get('synapses', {}):
                        syn_cell_id = int(syn_cell_id_str)
                        index.setdefault(syn_cell_id, []).append((node_id, cell_id_int, seg_idx))
        return index

    def activate_cells(self, active_columns: Set[int], predictive_cells_t: Set[int]) -> Tuple[Set[int], float]:
        active_cells, predicted_cells_in_active_cols = set(), 0
        for col_id in active_columns:
            valid_cids = []
            for cid in self.graph.nodes[col_id]['cells'].keys():
                try:
                    valid_cids.append(int(cid))
                except (ValueError, TypeError):
                    continue
            cells_in_column = set(valid_cids)

            predicted_cells_in_col = cells_in_column.intersection(predictive_cells_t)
            if predicted_cells_in_col:
                active_cells.update(predicted_cells_in_col)
                predicted_cells_in_active_cols += len(predicted_cells_in_col)
            else:
                active_cells.update(cells_in_column)
        bursting_cells = len(active_cells) - predicted_cells_in_active_cols
        surprise = (bursting_cells / len(active_cells)) if active_cells else 0.0
        return active_cells, surprise

    def learn_on_dendrites(self, active_cells: Set[int], predictive_cells_t: Set[int],
                           active_cells_t_minus_1: Set[int]):
        for cell_id in active_cells.intersection(predictive_cells_t):
            self._reinforce_active_segments(cell_id, active_cells_t_minus_1)
        unpredicted_active_cells = active_cells - predictive_cells_t
        if active_cells_t_minus_1 and unpredicted_active_cells:
            for cell_id in unpredicted_active_cells:
                self._grow_new_segment(cell_id, active_cells_t_minus_1)

    def _reinforce_active_segments(self, cell_id: int, active_cells_t_minus_1: Set[int]):
        node_id = cell_id // CELL_ID_MULTIPLIER
        if node_id not in self.graph.nodes: return

        cells_dict = self.graph.nodes[node_id]['cells']
        if not isinstance(cells_dict, dict): return

        cell_data = cells_dict.get(str(cell_id))
        if not isinstance(cell_data, dict): return

        for segment in cell_data.get('dendrites', []):
            if isinstance(segment, dict) and segment.get('was_active', False):
                for syn_id, perm in segment.get('synapses', {}).items():
                    change = self.network_model.permanence_increment if int(
                        syn_id) in active_cells_t_minus_1 else -self.network_model.permanence_decrement
                    segment['synapses'][syn_id] = min(1.0, max(0.0, perm + change))
                segment['was_active'] = False

    def _grow_new_segment(self, cell_id: int, active_cells_t_minus_1: Set[int]):
        node_id = cell_id // CELL_ID_MULTIPLIER
        if node_id not in self.graph.nodes: return

        cells_dict = self.graph.nodes[node_id]['cells']
        if not isinstance(cells_dict, dict): return

        cell_id_str = str(cell_id)

        cell_data = cells_dict.get(cell_id_str)
        if not isinstance(cell_data, dict):
            cell_data = {'dendrites': []}
            cells_dict[cell_id_str] = cell_data

        k = min(len(active_cells_t_minus_1), self.network_model.activation_threshold)
        if k <= 0: return

        sample = random.sample(list(active_cells_t_minus_1), k=k)
        new_segment = {'synapses': {str(pid): self.network_model.initial_permanence for pid in sample},
                       'was_active': False}

        dendrites = cell_data.get('dendrites', [])
        if not isinstance(dendrites, list):
            dendrites = []

        dendrites.append(new_segment)
        cell_data['dendrites'] = dendrites

        new_seg_idx = len(dendrites) - 1
        for syn_cell_id in sample:
            self.dendrite_index.setdefault(syn_cell_id, []).append((node_id, cell_id, new_seg_idx))

    def calculate_predictions(self, active_cells: Set[int]) -> Set[int]:
        predictive_cells, activated_segments = set(), set()
        for active_cell_id in active_cells:
            if active_cell_id in self.dendrite_index:
                for node_id, cell_id, seg_idx in self.dendrite_index[active_cell_id]:
                    if (node_id, cell_id, seg_idx) in activated_segments: continue
                    if node_id not in self.graph.nodes: continue

                    cells_dict = self.graph.nodes[node_id]['cells']
                    if not isinstance(cells_dict, dict): continue

                    cell_data = cells_dict.get(str(cell_id))
                    if not isinstance(cell_data, dict): continue

                    dendrites = cell_data.get('dendrites', [])
                    if not isinstance(dendrites, list) or seg_idx >= len(dendrites): continue

                    segment = dendrites[seg_idx]
                    if not isinstance(segment, dict): continue

                    segment['was_active'] = False
                    connected_syns = {int(s) for s, p in segment.get('synapses', {}).items() if
                                      p >= self.network_model.connected_permanence}
                    if len(connected_syns.intersection(active_cells)) >= self.network_model.activation_threshold:
                        predictive_cells.add(cell_id)
                        segment['was_active'] = True
                        activated_segments.add((node_id, cell_id, seg_idx))
        return predictive_cells


class GNGLearner:
    def __init__(self, graph: nx.Graph, network_model: BrainNetwork, sdr_helper: SDRHelper):
        self.graph = graph
        self.network_model = network_model
        self.sdr_helper = sdr_helper

    def update_spatial_topology(self, s1_id: int, s2_id: int, input_sdr: Set[int], iteration_count: int,
                                adaptive_lr: float) -> int:
        nodes_added = 0
        self._adapt_winner(s1_id, input_sdr, adaptive_lr)
        self._add_or_refresh_edge(s1_id, s2_id)
        self._age_and_prune_edges(s1_id)
        for _, data in self.graph.nodes(data=True):
            if 'error' in data:
                data['error'] *= self.network_model.error_decay_rate
        if iteration_count > 0 and iteration_count % self.network_model.n_iter_before_neuron_added == 0:
            if self._grow_network(iteration_count):
                nodes_added = 1
        return nodes_added

    def _adapt_winner(self, winner_id: int, input_sdr: Set[int], adaptive_lr: float):
        winner_sdr = self.graph.nodes[winner_id]['prototype_sdr']
        overlap = self.sdr_helper.overlap(input_sdr, winner_sdr)
        dissimilarity = 1.0 - (overlap / self.sdr_helper.sparsity)
        self.graph.nodes[winner_id]['error'] += dissimilarity

        self._adapt_node_sdr(winner_id, input_sdr, adaptive_lr)
        neighbor_lr = adaptive_lr * self.network_model.neighbor_learning_rate / self.network_model.winner_learning_rate
        for neighbor_id in self.graph.neighbors(winner_id):
            if self.graph.nodes[neighbor_id].get('type') == 'sensory':
                self._adapt_node_sdr(neighbor_id, input_sdr, neighbor_lr)

    def _adapt_node_sdr(self, node_id: int, target_sdr: Set[int], learning_rate: float):
        node_sdr = self.graph.nodes[node_id]['prototype_sdr']
        num_to_change = int(learning_rate * self.sdr_helper.sparsity)
        if num_to_change == 0: return
        add_candidates = list(target_sdr - node_sdr)
        if add_candidates:
            node_sdr.update(random.sample(add_candidates, min(len(add_candidates), num_to_change)))
        while len(node_sdr) > self.sdr_helper.sparsity:
            node_sdr.remove(random.choice(list(node_sdr - target_sdr) or list(node_sdr)))

    def _add_or_refresh_edge(self, u: int, v: int):
        if u != v: self.graph.add_edge(u, v, age=0)

    def _age_and_prune_edges(self, winner_id: int):
        for u, v, data in list(self.graph.edges(winner_id, data=True)):
            data['age'] += 1
            if data['age'] > self.network_model.max_edge_age:
                self.graph.remove_edge(u, v)

    def _grow_network(self, iteration_count: int) -> bool:
        sensory_nodes = [nid for nid, data in self.graph.nodes(data=True) if data.get('type') == 'sensory']
        if len(sensory_nodes) < 2: return False

        try:
            q_id = max(sensory_nodes, key=lambda nid: self.graph.nodes[nid].get('error', 0))
            q_node = self.graph.nodes[q_id]

            neighbors = [n for n in self.graph.neighbors(q_id) if n in sensory_nodes]
            if not neighbors: return False
            f_id = max(neighbors, key=lambda nid: self.graph.nodes[nid].get('error', 0))
            f_node = self.graph.nodes[f_id]
        except (ValueError, IndexError):
            return False

        new_node_id = max([n for n in self.graph.nodes if n > 0] + [0]) + 1
        new_sdr_union = q_node['prototype_sdr'].union(f_node['prototype_sdr'])
        new_sdr = set(random.sample(list(new_sdr_union), self.sdr_helper.sparsity))
        cells = {(new_node_id * CELL_ID_MULTIPLIER + i): {'dendrites': []} for i in
                 range(self.network_model.cells_per_column)}
        self.graph.add_node(new_node_id, prototype_sdr=new_sdr, error=(q_node['error'] + f_node['error']) / 2,
                            cells=cells, last_active_iter=iteration_count, type='sensory')

        if self.graph.has_edge(q_id, f_id): self.graph.remove_edge(q_id, f_id)
        self._add_or_refresh_edge(q_id, new_node_id)
        self._add_or_refresh_edge(f_id, new_node_id)
        q_node['error'] *= 0.5;
        f_node['error'] *= 0.5
        return True


class VisualCortex:
    def __init__(self, sdr_helper: SDRHelper):
        self.sdr_helper = sdr_helper
        self.num_features = 5

    def to_sdr(self, raw_obs: np.ndarray, timestep: int) -> Set[int]:
        grid = raw_obs.reshape((int(math.sqrt(raw_obs.size)), -1))
        grid_size = grid.shape[0]
        segment_size = self.sdr_helper.dimensionality // self.num_features
        sdr = set()
        player_pos = self._find_pos(grid, 0.5)
        goal_pos = self._find_pos(grid, 1.0)
        if player_pos: sdr.update(
            self._encode_scalar(player_pos[0] * grid_size + player_pos[1], grid_size * grid_size, 0, segment_size))
        if goal_pos: sdr.update(
            self._encode_scalar(goal_pos[0] * grid_size + goal_pos[1], grid_size * grid_size, 1, segment_size))
        if player_pos and goal_pos:
            vec = (goal_pos[0] - player_pos[0], goal_pos[1] - player_pos[1])
            sdr.update(self._encode_vector(vec, grid_size, 3, segment_size))
        sdr.update(self._encode_scalar(timestep % 100, 100, 4, segment_size))
        return sdr

    def _encode_scalar(self, value, max_value, feature_idx, segment_size, salt=0):
        base_offset = feature_idx * segment_size
        rng = random.Random(hash((value, max_value, salt)))
        num_bits = max(2, self.sdr_helper.sparsity // 8)
        return {base_offset + i for i in rng.sample(range(segment_size), num_bits)}

    def _encode_vector(self, vec, grid_size, feature_idx, segment_size):
        x_seg_size = y_seg_size = segment_size // 2
        sdr = set()
        sdr.update(self._encode_scalar(vec[0] + grid_size, grid_size * 2, feature_idx, x_seg_size, salt='x'))
        sdr.update(self._encode_scalar(vec[1] + grid_size, grid_size * 2, feature_idx, y_seg_size, salt='y'))
        return sdr

    @staticmethod
    def _find_pos(grid: np.ndarray, value: float) -> Union[Tuple[int, int], None]:
        pos_arr = np.argwhere(grid == value)
        return tuple(pos_arr[0]) if pos_arr.size > 0 else None


class SDRIndex:
    def __init__(self, dimensionality: int):
        self.dim = dimensionality
        if faiss:
            self.index = faiss.IndexFlatL2(dimensionality)
        else:
            self.index = None
        self.prototype_matrix = None
        self.node_ids: List[int] = []
        self.needs_rebuild = True
        self.nodes_added_since_rebuild = 0
        self.rebuild_threshold = 10

    @staticmethod
    @jax.jit
    def _jax_batch_overlap(query_vector, prototype_matrix):
        scores = jnp.dot(prototype_matrix, query_vector)
        k = jnp.minimum(2, prototype_matrix.shape[0])
        top_scores, top_indices = jax.lax.top_k(scores, k=k)
        return top_scores, top_indices

    def build_if_needed(self, graph: nx.Graph, sdr_helper: SDRHelper):
        if not self.needs_rebuild: return
        sensory_nodes = [(nid, data['prototype_sdr']) for nid, data in graph.nodes(data=True) if
                         data.get('type') == 'sensory']
        if not sensory_nodes: return
        self.node_ids, sdr_sets = zip(*sensory_nodes)
        if self.index:
            sdr_vectors_np = np.zeros((len(sdr_sets), self.dim), dtype=np.float32)
            for i, sdr in enumerate(sdr_sets):
                sdr_vectors_np[i, sdr_helper.sdr_to_list(sdr)] = 1.0
            self.index.reset()
            self.index.add(sdr_vectors_np)
        if JAX_AVAILABLE:
            sdr_vectors_jnp = jnp.zeros((len(sdr_sets), self.dim), dtype=jnp.float32)
            for i, sdr in enumerate(sdr_sets):
                indices = jnp.array(sdr_helper.sdr_to_list(sdr))
                sdr_vectors_jnp = sdr_vectors_jnp.at[i, indices].set(1.0)
            self.prototype_matrix = sdr_vectors_jnp
        self.needs_rebuild = False
        self.nodes_added_since_rebuild = 0

    def search(self, query_sdr: Set[int], k: int, graph: nx.Graph, sdr_helper: SDRHelper) -> List[Tuple[int, float]]:
        self.build_if_needed(graph, sdr_helper)
        if self.index and self.index.ntotal > 0:
            query_emb = np.zeros(self.dim, dtype=np.float32)
            query_emb[sdr_helper.sdr_to_list(query_sdr)] = 1.0
            distances, indices = self.index.search(query_emb.reshape(1, -1), k)
            return [(self.node_ids[idx], self.index.d - dist) for idx, dist in zip(indices[0], distances[0]) if
                    0 <= idx < len(self.node_ids)]
        if JAX_AVAILABLE and self.prototype_matrix is not None and self.prototype_matrix.shape[0] > 0:
            query_vector = jnp.zeros(self.dim, dtype=jnp.float32).at[jnp.array(sdr_helper.sdr_to_list(query_sdr))].set(
                1.0)
            scores, indices = self._jax_batch_overlap(query_vector, self.prototype_matrix)
            return [(self.node_ids[idx], float(score)) for score, idx in zip(scores, indices)]
        return self._manual_search(query_sdr, k, graph, sdr_helper)

    @staticmethod
    def _manual_search(query_sdr: Set[int], k: int, graph: nx.Graph, sdr_helper: SDRHelper) -> List[Tuple[int, int]]:
        sensory_nodes = [nid for nid, data in graph.nodes(data=True) if data.get('type') == 'sensory']
        if not sensory_nodes: return []
        scores = [(nid, sdr_helper.overlap(query_sdr, graph.nodes[nid]['prototype_sdr'])) for nid in sensory_nodes]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class MotorCortex:
    def __init__(self, network_model: BrainNetwork):
        self.action_nodes: Dict[str, int] = {}
        self.epsilon = network_model.rl_exploration_rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9998

    def define_actions(self, action_names: List[str], graph: nx.Graph, sdr_helper: SDRHelper):
        for name in action_names:
            if name not in self.action_nodes:
                sdr = sdr_helper.encode_token(f"action_{name}")
                node_id = -(len(self.action_nodes) + 1)
                graph.add_node(node_id, prototype_sdr=sdr, type='action', action_name=name)
                self.action_nodes[name] = node_id

    def select_action(self, state_node_id: int, graph: nx.Graph, q_table: Dict) -> str:
        action_names = list(self.action_nodes.keys())
        if not action_names: raise ValueError("No actions defined.")
        if random.random() < self.epsilon: return random.choice(action_names)
        cand_actions = [n for n in graph.neighbors(state_node_id) if graph.nodes[n].get('type') == 'action']
        if not cand_actions: return random.choice(action_names)
        q_values = {act: q_table.get((state_node_id, act), 0.0) for act in cand_actions}
        best_action_node = max(q_values, key=q_values.get)
        return graph.nodes[best_action_node]['action_name']

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay


class RewardSystem:
    def __init__(self, network_model: BrainNetwork):
        self.network_model = network_model
        self.alpha = 0.1
        self.orig_inc = network_model.permanence_increment
        self.orig_dec = network_model.permanence_decrement

    def apply(self, reward: float):
        reward = max(-1.0, min(1.0, reward))
        self.network_model.permanence_increment = self.orig_inc * (1.0 + self.alpha * reward)
        self.network_model.permanence_decrement = self.orig_dec * (1.0 - self.alpha * reward)


class ControlLearner:
    def __init__(self, service_ref: 'STAGNetworkService'):
        self.service = service_ref
        self.visual_cortex = VisualCortex(service_ref.sdr_helper)
        self.motor_cortex = MotorCortex(service_ref.network_model)
        self.reward_system = RewardSystem(service_ref.network_model)
        self.env: BaseEnvironment = None
        self.intrinsic_reward_factor = 0.05

    def setup_environment(self, env: BaseEnvironment):
        self.env = env
        self.motor_cortex.define_actions(env.action_space, self.service.graph, self.service.sdr_helper)
        self.service.ensure_graph_initialized()

    def train_online(self, episodes: int, max_steps: int) -> Generator[Dict, None, None]:
        if not self.env: raise RuntimeError("Environment not set up.")

        for ep in range(episodes):
            obs = self.env.reset().observation
            total_reward, total_q_diff = 0.0, 0.0
            progress_bar = tqdm(range(max_steps), desc=f"Episode {ep + 1}/{episodes}", leave=False)
            state_node_id = None

            for step in progress_bar:
                state_sdr = self.visual_cortex.to_sdr(obs, step)
                surprise, graph_changed = self.service.process_sdr(state_sdr)
                winners = self.service.sdr_index.search(state_sdr, 1, self.service.graph, self.service.sdr_helper)
                if not winners: continue
                state_node_id = winners[0][0]

                # --- REFACTORED: Broadcast graph update periodically or on change ---
                if step % 50 == 0 or graph_changed:
                    self.service._broadcast_graph_update(state_node_id)
                else:
                    self.service._broadcast_activation(state_node_id)

                action_name = self.motor_cortex.select_action(state_node_id, self.service.graph,
                                                              self.service.action_q_table)
                action_node_id = self.motor_cortex.action_nodes[action_name]
                env_state = self.env.step(action_name)
                new_obs, reward, done = env_state.observation, env_state.reward, env_state.done
                total_reward += reward
                shaped_reward = reward + (surprise * self.intrinsic_reward_factor)
                self.reward_system.apply(shaped_reward)
                new_state_sdr = self.visual_cortex.to_sdr(new_obs, step + 1)
                self.service.process_sdr(new_state_sdr)
                new_winners = self.service.sdr_index.search(new_state_sdr, 1, self.service.graph,
                                                            self.service.sdr_helper)
                new_state_node_id = new_winners[0][0] if new_winners else None
                q_diff = self._update_q_value(state_node_id, action_node_id, shaped_reward, new_state_node_id, done)
                total_q_diff += abs(q_diff)

                step_data_for_viz = {
                    'event_type': 'step_update',
                    'observation': new_obs.tolist(),
                    'episode': ep + 1, 'step': step, 'action': action_name,
                    'reward': reward, 'total_reward': total_reward, 'done': done,
                }
                self.service._broadcast_event(step_data_for_viz)

                yield {
                    'type': 'step', 'episode': ep + 1, 'step': step, 'reward': reward,
                    'total_reward': total_reward, 'done': done
                }
                obs = new_obs
                if done: break

            progress_bar.close()
            # Broadcast the final graph state for the episode
            if state_node_id is not None:
                self.service._broadcast_graph_update(state_node_id)

            self.motor_cortex.decay_epsilon()
            self.service.stability_tracker.track_q_volatility(total_q_diff / (step + 1 if step > 0 else 1))

            yield {
                'type': 'episode_end', 'episode': ep + 1, 'reward': total_reward,
                'steps': step + 1, 'epsilon': self.motor_cortex.epsilon
            }
            if (ep + 1) % 250 == 0:
                self.service.create_snapshot(f"train_{self.env.name}_ep_{ep + 1}")

    def _update_q_value(self, state_id, action_id, reward, next_state_id, done) -> float:
        q_table = self.service.action_q_table
        lr = self.service.stability_tracker.get_adaptive_rl_lr()
        gamma = self.service.network_model.rl_discount_factor
        current_q = q_table.get((state_id, action_id), 0.0)
        max_next_q = 0.0
        if not done and next_state_id:
            next_action_nodes = [n for n in self.service.graph.neighbors(next_state_id) if
                                 self.service.graph.nodes[n].get('type') == 'action']
            if next_action_nodes: max_next_q = max(
                [q_table.get((next_state_id, next_action), 0.0) for next_action in next_action_nodes], default=0.0)
        target = reward + gamma * max_next_q
        new_q = current_q + lr * (target - current_q)
        q_table[(state_id, action_id)] = new_q
        self.service.graph.add_edge(state_id, action_id, age=0)
        if next_state_id: self.service.graph.add_edge(action_id, next_state_id, age=0)
        return new_q - current_q


class STAGNetworkService:
    """Main service class orchestrating the STAG network's learning components."""

    def __init__(self, network_id: int):
        self.network_model = BrainNetwork.objects.get(pk=network_id)
        self.sdr_helper = SDRHelper(self.network_model.sdr_dimensionality, self.network_model.sdr_sparsity)
        self.graph = nx.Graph()
        self.iteration_count = 0
        self.is_training = False
        self.active_cells_t_minus_1: Set[int] = set()
        self.predictive_cells_t: Set[int] = set()
        self.action_q_table: Dict[Tuple[int, int], float] = {}
        self.stability_tracker = StabilityTracker(self.network_model)
        self.htm_learner = HTMLearner(self.graph, self.network_model)
        self.gng_learner = GNGLearner(self.graph, self.network_model, self.sdr_helper)
        self.control_learner = ControlLearner(self)
        self.sdr_index = SDRIndex(self.sdr_helper.dimensionality)
        self.channel_layer = get_channel_layer()
        self.channel_group_name = f'brain_training_{self.network_model.id}'
        self.graph_has_changed = False
        self._load_latest_snapshot()

    def _get_sensory_nodes(self) -> List[int]:
        return [nid for nid, data in self.graph.nodes(data=True) if data.get('type') == 'sensory']

    def _load_latest_snapshot(self):
        latest_snapshot = GraphSnapshot.objects.filter(network=self.network_model).order_by('-created_at').first()
        if latest_snapshot:
            self.load_from_snapshot(latest_snapshot.id)
        else:
            print("No snapshots found. Starting with a blank graph.")

    def _broadcast_event(self, payload: Dict):
        if self.channel_layer:
            message = {'type': 'broadcast_event', 'payload': payload}
            async_to_sync(self.channel_layer.group_send)(self.channel_group_name, message)

    def _broadcast_graph_update(self, active_node_id: int):
        """Broadcasts the entire graph state."""
        graph_state = self.get_graph_state_for_viz()
        self._broadcast_event({
            'event_type': 'graph_update',
            'graph_state': graph_state,
            'active_node_id': active_node_id
        })
        self.graph_has_changed = False  # Reset flag after sending

    def _broadcast_activation(self, active_node_id: int):
        """Broadcasts only the ID of the active node for efficiency."""
        self._broadcast_event({
            'event_type': 'activation_update',
            'active_node_id': active_node_id
        })

    def process_sdr(self, sdr: Set[int]) -> Tuple[float, bool]:
        if not self.is_training: return 0.0, False
        graph_changed_this_step = False
        winners = self.sdr_index.search(sdr, 2, self.graph, self.sdr_helper)
        if not winners:
            self._add_node(sdr=sdr)
            self.graph_has_changed = True
            self.sdr_index.nodes_added_since_rebuild += 1
            if self.sdr_index.nodes_added_since_rebuild >= self.sdr_index.rebuild_threshold:
                self.sdr_index.needs_rebuild = True
            return 1.0, True

        s1_id = winners[0][0]
        self.graph.nodes[s1_id]['last_active_iter'] = self.iteration_count
        active_cells, surprise = self.htm_learner.activate_cells({s1_id}, self.predictive_cells_t)
        self.htm_learner.learn_on_dendrites(active_cells, self.predictive_cells_t, self.active_cells_t_minus_1)
        self.predictive_cells_t = self.htm_learner.calculate_predictions(active_cells)
        self.active_cells_t_minus_1 = active_cells

        s2_id = winners[1][0] if len(winners) > 1 else s1_id
        adaptive_gng_lr = self.stability_tracker.get_adaptive_gng_lr()
        nodes_added = self.gng_learner.update_spatial_topology(s1_id, s2_id, sdr, self.iteration_count, adaptive_gng_lr)

        if nodes_added > 0:
            self.graph_has_changed = True
            self.sdr_index.nodes_added_since_rebuild += nodes_added
            if self.sdr_index.nodes_added_since_rebuild >= self.sdr_index.rebuild_threshold:
                self.sdr_index.needs_rebuild = True
            self.stability_tracker.track_graph_growth(nodes_added)
            graph_changed_this_step = True

        self.iteration_count += 1
        return surprise, graph_changed_this_step

    def ensure_graph_initialized(self):
        if len(self._get_sensory_nodes()) < 2:
            sdr1 = self.sdr_helper._create_random_sdr()
            sdr2 = self.sdr_helper._create_random_sdr()
            node1_id = self._add_node(sdr=sdr1)
            node2_id = self._add_node(sdr=sdr2)
            self.graph.add_edge(node1_id, node2_id, age=0)
            self.graph_has_changed = True
            self.sdr_index.needs_rebuild = True
            print(f"Graph initialized with sensory nodes {node1_id} and {node2_id}.")

    def _add_node(self, sdr: Set[int], error: float = 0.1) -> int:
        node_id = max([n for n in self.graph.nodes if n > 0] + [0]) + 1
        cells = {(node_id * CELL_ID_MULTIPLIER + i): {'dendrites': []} for i in
                 range(self.network_model.cells_per_column)}
        self.graph.add_node(node_id, prototype_sdr=sdr, error=error, cells=cells, last_active_iter=self.iteration_count,
                            type='sensory')
        return node_id

    def train_on_env(self, env: BaseEnvironment, episodes: int, max_steps: int) -> Generator[Dict, None, None]:
        self.start_training_session()
        try:
            self.control_learner.setup_environment(env)
            yield from self.control_learner.train_online(episodes, max_steps)
        finally:
            self.end_training_session(snapshot_name_prefix=f"post_env_{env.name}")

    def start_training_session(self):
        self.is_training = True

    def end_training_session(self, create_snapshot=True, snapshot_name_prefix="manual_snapshot"):
        self.is_training = False
        if create_snapshot:
            self.create_snapshot(f"{snapshot_name_prefix}_iter_{self.iteration_count}")

    @transaction.atomic
    def create_snapshot(self, name: str) -> Dict[str, Any]:
        if GraphSnapshot.objects.filter(network=self.network_model, name=name).exists():
            name = f"{name}_{int(time.time())}"
        snapshot = GraphSnapshot.objects.create(network=self.network_model, name=name)
        SnapshotNeuron.objects.bulk_create([
            SnapshotNeuron(snapshot=snapshot, neuron_id=nid,
                           prototype_sdr=self.sdr_helper.sdr_to_list(data['prototype_sdr']),
                           error=data.get('error', 0.0), cells=data.get('cells', {}),
                           last_active_iter=data.get('last_active_iter', 0),
                           node_type=data.get('type', 'sensory'), action_name=data.get('action_name'))
            for nid, data in self.graph.nodes(data=True)], batch_size=500)
        SnapshotEdge.objects.bulk_create([
            SnapshotEdge(snapshot=snapshot, source_id=u, target_id=v, age=data.get('age', 0))
            for u, v, data in self.graph.edges(data=True)], batch_size=500)
        self.graph_has_changed = False
        return GraphSnapshotDetailSerializer(snapshot).data

    @transaction.atomic
    def load_from_snapshot(self, snapshot_id: int):
        snapshot = GraphSnapshot.objects.select_related('network').get(pk=snapshot_id)
        self.graph.clear()
        for neuron_data in SnapshotNeuron.objects.filter(snapshot=snapshot):
            self.graph.add_node(neuron_data.neuron_id,
                                prototype_sdr=self.sdr_helper.list_to_sdr(neuron_data.prototype_sdr),
                                error=neuron_data.error, cells=neuron_data.cells,
                                last_active_iter=neuron_data.last_active_iter,
                                type=neuron_data.node_type, action_name=neuron_data.action_name)
        for edge_data in SnapshotEdge.objects.filter(snapshot=snapshot):
            self.graph.add_edge(edge_data.source_id, edge_data.target_id, age=edge_data.age)
        self.htm_learner = HTMLearner(self.graph, self.network_model)
        self.sdr_index.needs_rebuild = True
        self.graph_has_changed = True
        print(f"STAG graph loaded from snapshot '{snapshot.name}'.")

    def get_graph_state_for_viz(self) -> Dict[str, List]:
        return {
            "nodes": [{'id': nid, 'type': d.get('type', 'sensory'), 'error': d.get('error', 0.0),
                       'action_name': d.get('action_name')}
                      for nid, d in self.graph.nodes(data=True)],
            "links": [{"source": u, "target": v} for u, v in self.graph.edges()]
        }
