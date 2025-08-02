import random
import math
import networkx as nx
import numpy as np
from typing import List, Set, Tuple, Dict, Any
from collections import Counter, deque

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from pybloom_live import ScalableBloomFilter

try:
    import faiss
except ImportError:
    faiss = None

from .models import BrainNetwork, GraphSnapshot
from .serializers import GraphSnapshotDetailSerializer
from .environments import BaseEnvironment, EnvironmentConfig, EnvironmentManager

# --- Constants ---
SNAPSHOT_SCHEMA_VERSION = "1.3"
CELL_ID_MULTIPLIER = 10000


# --- Helper and Modular Classes ---

class SDRHelper:
    def __init__(self, dimensionality: int, sparsity: int):
        self.dimensionality = dimensionality
        self.sparsity = sparsity
        self.encoder_cache: Dict[str, Set[int]] = {}
        self.bloom = ScalableBloomFilter(error_rate=0.001)

    def encode_token(self, token: str) -> Set[int]:
        if token not in self.encoder_cache:
            sdr = self.create_random_sdr()
            self.encoder_cache[token] = sdr
            for bit in sdr: self.bloom.add(str(bit))
        return self.encoder_cache[token]

    def create_random_sdr(self) -> Set[int]:
        if self.sparsity > self.dimensionality: raise ValueError("Sparsity cannot be greater than dimensionality.")
        return set(random.sample(range(self.dimensionality), self.sparsity))

    def union(self, sdr_list: List[Set[int]]) -> Set[int]:
        result = set();
        [result.update(sdr) for sdr in sdr_list];
        return result

    def overlap(self, sdr1: Set[int], sdr2: Set[int]) -> int:
        return len(sdr1.intersection(sdr2))

    def sdr_to_list(self, sdr: Set[int]) -> List[int]:
        return sorted(list(sdr))

    def list_to_sdr(self, sdr_list: List[int]) -> Set[int]:
        return set(sdr_list)


class HTMLearner:
    def __init__(self, service_ref):
        self.service = service_ref
        self.graph = service_ref.graph
        self.network_model = service_ref.network_model
        self.dendrite_index: Dict[int, List[Tuple[int, int, int]]] = {}

    def rebuild_dendrite_index(self):
        self.dendrite_index.clear()
        for node_id, node_data in self.graph.nodes(data=True):
            for cell_id, cell_data in node_data.get('cells', {}).items():
                for seg_idx, segment in enumerate(cell_data.get('dendrites', [])):
                    for syn_cell_id in segment.get('synapses', {}):
                        syn_cell_id = int(syn_cell_id)
                        if syn_cell_id not in self.dendrite_index: self.dendrite_index[syn_cell_id] = []
                        self.dendrite_index[syn_cell_id].append((node_id, cell_id, seg_idx))

    def activate_cells(self, active_columns: Set[int]) -> Set[int]:
        active_cells, predicted_cells_in_active_cols = set(), 0
        sorted_active_columns = sorted(active_columns, key=lambda c: self.graph.nodes[c].get('last_active_iter', 0),
                                       reverse=True)[:100]
        for col_id in sorted_active_columns:
            column_node = self.graph.nodes[col_id]
            cells_in_column = set(column_node['cells'].keys())
            predicted_cells_in_col = cells_in_column.intersection(self.service.predictive_cells_t)
            if predicted_cells_in_col:
                active_cells.update(predicted_cells_in_col)
                predicted_cells_in_active_cols += len(predicted_cells_in_col)
            else:
                active_cells.update(cells_in_column)
        bursting_cells = len(active_cells) - predicted_cells_in_active_cols
        self.service.metrics['surprise'] = bursting_cells / len(active_cells) if active_cells else 0.0
        return active_cells

    def learn_on_dendrites(self, active_cells: Set[int]):
        for cell_id in active_cells.intersection(self.service.predictive_cells_t):
            node_id = self.service._get_node_id_for_cell(cell_id)
            if node_id is None: continue
            cell_data = self.graph.nodes[node_id]['cells'][cell_id]
            for segment in cell_data['dendrites']:
                if not segment.get('was_active', False): continue
                for syn_cell_id, perm in segment['synapses'].items():
                    if int(syn_cell_id) in self.service.active_cells_t_minus_1:
                        segment['synapses'][syn_cell_id] = min(1.0, perm + self.network_model.permanence_increment)
                    else:
                        segment['synapses'][syn_cell_id] = max(0.0, perm - self.network_model.permanence_decrement)
                segment['was_active'] = False
        unpredicted_active_cells = active_cells - self.service.predictive_cells_t
        if self.service.active_cells_t_minus_1 and unpredicted_active_cells:
            for cell_id in unpredicted_active_cells:
                node_id = self.service._get_node_id_for_cell(cell_id)
                if node_id is None or node_id not in self.graph.nodes: continue
                cell_data = self.graph.nodes[node_id]['cells'][cell_id]
                new_segment = {'synapses': {}, 'was_active': False}
                num_synapses_to_form = self.network_model.activation_threshold
                k = min(len(self.service.active_cells_t_minus_1), num_synapses_to_form)
                if k > 0:
                    sample = random.sample(list(self.service.active_cells_t_minus_1), k=k)
                    for prev_cell_id in sample: new_segment['synapses'][
                        str(prev_cell_id)] = self.network_model.initial_permanence
                cell_data['dendrites'].append(new_segment)
                new_seg_idx = len(cell_data['dendrites']) - 1
                for syn_cell_id in new_segment['synapses']:
                    syn_cell_id = int(syn_cell_id)
                    if syn_cell_id not in self.dendrite_index: self.dendrite_index[syn_cell_id] = []
                    self.dendrite_index[syn_cell_id].append((node_id, cell_id, new_seg_idx))

    def calculate_predictions(self, active_cells: Set[int]) -> Set[int]:
        predictive_cells, activated_segments = set(), set()
        for active_cell_id in active_cells:
            if active_cell_id in self.dendrite_index:
                for node_id, cell_id, seg_idx in self.dendrite_index[active_cell_id]:
                    if (node_id, cell_id, seg_idx) in activated_segments: continue
                    segment = self.graph.nodes[node_id]['cells'][cell_id]['dendrites'][seg_idx]
                    segment['was_active'] = False
                    connected_synapses = {int(s) for s, p in segment['synapses'].items() if
                                          p >= self.network_model.connected_permanence}
                    if len(connected_synapses.intersection(active_cells)) >= self.network_model.activation_threshold:
                        predictive_cells.add(cell_id)
                        segment['was_active'] = True
                        activated_segments.add((node_id, cell_id, seg_idx))
        return predictive_cells


class GNGLearner:
    def __init__(self, service_ref):
        self.service = service_ref
        self.graph = service_ref.graph
        self.network_model = service_ref.network_model
        self.sdr_helper = service_ref.sdr_helper

    def update_spatial_topology(self, s1_id: int, s2_id: int, input_sdr: Set[int]):
        self.service._add_edge(s1_id, s2_id, age=0)
        s1_overlap = self.sdr_helper.overlap(input_sdr, self.graph.nodes[s1_id]['prototype_sdr'])
        dissimilarity = 1.0 - (s1_overlap / self.sdr_helper.sparsity)
        self.graph.nodes[s1_id]['error'] += dissimilarity
        decay_factor = 1 / (1 + self.service.iteration_count / 10000)
        winner_lr, neighbor_lr = self.network_model.winner_learning_rate * decay_factor, self.network_model.neighbor_learning_rate * decay_factor
        self._adapt_node_sdr(s1_id, input_sdr, winner_lr)
        for neighbor_id in self.graph.neighbors(s1_id): self._adapt_node_sdr(neighbor_id, input_sdr, neighbor_lr)
        self._age_and_prune_edges(s1_id)
        max_nodes = getattr(self.network_model, 'max_nodes', 5000)
        if self.graph.number_of_nodes() < max_nodes and self.service.iteration_count > 0 and self.service.iteration_count % self.network_model.n_iter_before_neuron_added == 0:
            self._grow_network()
        for node_id, data in self.graph.nodes(data=True): data['error'] *= self.network_model.error_decay_rate

    def _adapt_node_sdr(self, node_id: int, input_sdr: Set[int], learning_rate: float):
        node_sdr = self.graph.nodes[node_id]['prototype_sdr']
        num_to_change = int(learning_rate * self.sdr_helper.sparsity)
        if num_to_change == 0: return
        add_candidates, remove_candidates = list(input_sdr - node_sdr), list(node_sdr - input_sdr)
        for _ in range(num_to_change):
            if add_candidates:
                bit_to_add = random.choice(add_candidates);
                node_sdr.add(bit_to_add);
                add_candidates.remove(bit_to_add)
        while len(node_sdr) > self.sdr_helper.sparsity:
            if remove_candidates:
                bit_to_remove = random.choice(remove_candidates)
                if bit_to_remove in node_sdr: node_sdr.remove(bit_to_remove)
                remove_candidates.remove(bit_to_remove)
            else:
                node_sdr.remove(random.choice(list(node_sdr)))
        self.graph.nodes[node_id]['prototype_sdr'] = node_sdr

    def _age_and_prune_edges(self, winner_id: int):
        edges_to_prune = []
        for u, v, data in self.graph.edges(winner_id, data=True):
            data['age'] += 1
            if data['age'] > self.network_model.max_edge_age: edges_to_prune.append((u, v))
        for u, v in edges_to_prune: self.service._remove_edge(u, v)
        stale_node_threshold = self.network_model.n_iter_before_neuron_added * 10
        for node_id in list(nx.isolates(self.graph)):
            last_active = self.graph.nodes[node_id].get('last_active_iter', 0)
            error = self.graph.nodes[node_id].get('error', 0)
            if (self.service.iteration_count - last_active > stale_node_threshold) and error < 0.01:
                self.service._remove_node(node_id)

    def _grow_network(self):
        if self.graph.number_of_nodes() < 2: return
        print("Attempting to grow the network...")
        nodes, errors = list(self.graph.nodes), [self.graph.nodes[n]['error'] for n in self.graph.nodes]
        total_error = sum(errors)
        if total_error <= 0: return
        q_id = random.choices(nodes, weights=errors, k=1)[0]
        q_neighbors = list(self.graph.neighbors(q_id))
        if not q_neighbors: return
        f_id = max(q_neighbors, key=lambda nid: self.graph.nodes[nid]['error'])
        interpolated_sdr = self.sdr_helper.union(
            [self.graph.nodes[q_id]['prototype_sdr'], self.graph.nodes[f_id]['prototype_sdr']])
        new_sdr = set(random.sample(list(interpolated_sdr), self.sdr_helper.sparsity))
        new_node_id = self.service._add_node(sdr=new_sdr, error=(self.graph.nodes[q_id]['error'] +
                                                                 self.graph.nodes[f_id]['error']) / 2)
        print(f"New neuron {new_node_id} grown near {q_id} and {f_id}.")
        self.service._remove_edge(q_id, f_id)
        self.service._add_edge(q_id, new_node_id)
        self.service._add_edge(f_id, new_node_id)
        self.graph.nodes[q_id]['error'] *= 0.5;
        self.graph.nodes[f_id]['error'] *= 0.5
        snapshot_name = f"auto_growth_iter_{self.service.iteration_count}_nodes_{self.graph.number_of_nodes()}"
        self.service.create_snapshot(snapshot_name)


class BookRLComponent:
    def __init__(self, service_ref):
        self.service = service_ref
        self.q_table = {}
        self.learning_rate = self.service.network_model.rl_learning_rate
        self.discount_factor = self.service.network_model.rl_discount_factor
        self.exploration_rate = self.service.network_model.rl_exploration_rate
        self.current_episode = []

    def start_reading_session(self):
        self.current_episode = []
        self.service.metrics['rl_episode'] = self.service.metrics.get('rl_episode', 0) + 1

    def observe_state(self, state_node_id: int, action_node_id: int):
        if self.current_episode:
            self.current_episode[-1]['next_state'] = state_node_id
        self.current_episode.append({'state': state_node_id, 'action': action_node_id, 'reward': 0, 'next_state': None})

    def choose_narrative_action(self, state_node_id: int) -> int:
        neighbors = list(self.service.graph.neighbors(state_node_id))
        if not neighbors: return state_node_id
        if random.random() < self.exploration_rate: return random.choice(neighbors)
        q_values = {neighbor: self.q_table.get((state_node_id, neighbor), 0) for neighbor in neighbors}
        return max(q_values, key=q_values.get)

    def apply_reward(self, reward: float):
        if self.current_episode: self.current_episode[-1]['reward'] = reward

    def update_q_values(self):
        for step in reversed(self.current_episode):
            if step['action'] is None or step['next_state'] is None: continue
            state, action, reward, next_state = step['state'], step['action'], step['reward'], step['next_state']
            next_neighbors = list(self.service.graph.neighbors(next_state))
            next_q_values = [self.q_table.get((next_state, a), 0) for a in next_neighbors]
            max_next_q = max(next_q_values) if next_q_values else 0
            current_q = self.q_table.get((state, action), 0)
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            self.q_table[(state, action)] = new_q
            if self.service.graph.has_edge(state, action):
                current_age = self.service.graph[state][action]['age']
                self.service.graph[state][action]['age'] = max(0, current_age - int(new_q * 10))

    def get_best_path(self, start_node: int, max_depth: int = 5) -> List[int]:
        path, current = [start_node], start_node
        for _ in range(max_depth):
            neighbors = list(self.service.graph.neighbors(current))
            if not neighbors: break
            next_node = max(neighbors, key=lambda n: self.q_table.get((current, n), 0))
            if next_node in path: break
            path.append(next_node);
            current = next_node
        return path


class VisualCortex:
    def __init__(self, service_ref):
        self.service = service_ref
        self.cnn = None

    def to_sdr(self, raw_obs: np.ndarray, k: int = None) -> Set[int]:
        emb = raw_obs.astype(np.float32).flatten()
        if len(emb) < self.service.sdr_helper.dimensionality:
            emb = np.pad(emb, (0, self.service.sdr_helper.dimensionality - len(emb)), 'constant')
        elif len(emb) > self.service.sdr_helper.dimensionality:
            emb = emb[:self.service.sdr_helper.dimensionality]
        max_val = emb.max()
        if max_val > 0: emb /= max_val
        if k is None: k = self.service.network_model.sdr_sparsity
        top_indices = np.argpartition(emb, -k)[-k:]
        return set(int(i) for i in top_indices)


class SDRIndex:
    def __init__(self, dimensionality: int):
        self.dim = dimensionality
        if faiss:
            self.index = faiss.IndexFlatL2(dimensionality)
        else:
            self.index = None; print("Warning: FAISS not found. Falling back to slower search.")
        self.node_ids: List[int] = []

    def add(self, node_id: int, sdr: Set[int]):
        if self.index:
            sdr_emb = np.zeros(self.dim, dtype=np.float32)
            sdr_emb[list(sdr)] = 1.0
            self.index.add(sdr_emb.reshape(1, -1))
            self.node_ids.append(node_id)

    def search(self, query_sdr: Set[int], k: int) -> List[int]:
        if self.index and self.index.ntotal > 0:
            query_emb = np.zeros(self.dim, dtype=np.float32)
            query_emb[list(query_sdr)] = 1.0
            _, I = self.index.search(query_emb.reshape(1, -1), k)
            return [self.node_ids[i] for i in I[0] if i < len(self.node_ids)]
        return []


class MotorCortex:
    def __init__(self, service_ref):
        self.service = service_ref
        self.action_nodes: Dict[str, int] = {}
        self.epsilon = self.service.network_model.rl_exploration_rate

    def define_actions(self, actions: List[str]):
        for action_name in actions:
            if action_name not in self.action_nodes:
                sdr = self.service.sdr_helper.create_random_sdr()
                node_id = -(len(self.action_nodes) + 1)
                self.service._add_node(sdr=sdr, node_type='action', action_name=action_name)
                self.action_nodes[action_name] = node_id

    def select_action(self, state_node_id: int) -> str:
        if random.random() < self.epsilon or not self.action_nodes:
            return random.choice(list(self.action_nodes.keys()))
        neighbors = list(self.service.graph.neighbors(state_node_id))
        candidate_actions = [n for n in neighbors if self.service.graph.nodes[n].get('type') == 'action']
        if not candidate_actions:
            return random.choice(list(self.action_nodes.keys()))
        best_action_node = min(candidate_actions, key=lambda n: self.service.graph[state_node_id][n].get('age', 999))
        return self.service.graph.nodes[best_action_node]['action_name']


class RewardSystem:
    def __init__(self, service_ref):
        self.service = service_ref
        self.alpha = 0.1  # Modulation strength

    def apply(self, reward: float):
        reward = max(-1, min(1, reward))
        if not hasattr(self, 'original_permanence_increment'):
            self.original_permanence_increment = self.service.network_model.permanence_increment
            self.original_permanence_decrement = self.service.network_model.permanence_decrement
        self.service.network_model.permanence_increment = self.original_permanence_increment * (1 + self.alpha * reward)
        self.service.network_model.permanence_decrement = self.original_permanence_decrement * (1 - self.alpha * reward)


class ControlLearner:
    def __init__(self, service_ref):
        self.service = service_ref
        self.visual_cortex = VisualCortex(service_ref)
        self.motor_cortex = MotorCortex(service_ref)
        self.reward_system = RewardSystem(service_ref)
        self.env = None

    def setup_environment(self, env: BaseEnvironment):
        self.env = env
        self.motor_cortex.define_actions(env.action_space)

    def train(self, episodes: int = 100, max_steps: int = 200):
        results = []
        for ep in range(episodes):
            env_state = self.env.reset()
            obs = env_state.observation
            total_reward = 0.0
            for step in range(max_steps):
                state_sdr = self.visual_cortex.to_sdr(obs)
                self.service.learn_from_sdr(state_sdr)
                winners = self.service._get_winners(state_sdr, 1)
                if not winners: continue
                state_node_id = winners[0][0]
                action = self.motor_cortex.select_action(state_node_id)
                action_node_id = self.motor_cortex.action_nodes[action]
                env_state = self.env.step(action)
                new_obs, reward, done = env_state.observation, env_state.reward, env_state.done
                self.reward_system.apply(reward)
                new_state_sdr = self.visual_cortex.to_sdr(new_obs)
                self._reinforce(state_sdr, action_node_id, new_state_sdr)
                obs = new_obs
                total_reward += reward
                if done: break
            results.append({'episode': ep + 1, 'reward': total_reward, 'steps': step + 1})
            print(f"Episode {ep + 1}/{episodes} finished after {step + 1} steps with reward: {total_reward}")
            if (ep + 1) % 10 == 0:
                self.service.create_snapshot(f"train_env_ep_{ep + 1}")
        return results

    def _reinforce(self, state_sdr: Set[int], action_node_id: int, new_state_sdr: Set[int]):
        state_winners = self.service._get_winners(state_sdr, 3)
        new_state_winners = self.service._get_winners(new_state_sdr, 3)
        for s_node, _ in state_winners:
            self.service._add_edge(s_node, action_node_id)
        for n_node, _ in new_state_winners:
            self.service._add_edge(action_node_id, n_node)


class STAGNetworkService:
    """
    Main service class that orchestrates all learning components.
    """

    def __init__(self, network_id: int):
        try:
            self.network_model = BrainNetwork.objects.get(pk=network_id)
            if self.network_model.n_iter_before_neuron_added <= 0:
                raise ValueError("n_iter_before_neuron_added must be a positive integer.")
        except BrainNetwork.DoesNotExist:
            raise ValueError(f"BrainNetwork with id {network_id} does not exist.")

        self.sdr_helper = SDRHelper(self.network_model.sdr_dimensionality, self.network_model.sdr_sparsity)
        self.graph = nx.Graph()
        self.iteration_count = 0

        self.htm_learner = HTMLearner(self)
        self.gng_learner = GNGLearner(self)
        self.rl_component = BookRLComponent(self)
        self.control_learner = ControlLearner(self)
        self.sdr_index = SDRIndex(self.sdr_helper.dimensionality)

        self.active_cells_t_minus_1: Set[int] = set()
        self.predictive_cells_t: Set[int] = set()

        self.channel_layer = get_channel_layer()
        self.channel_group_name = f'brain_{self.network_model.id}'

        self.graph_has_changed = False
        self.nodes_added_since_snapshot = 0

        self.metrics = {'surprise': 0.0, 'node_count': 0, 'edge_count': 0}

        self._load_latest_snapshot()

    def train_on_env(self, env: BaseEnvironment, episodes: int = 100, max_steps: int = 200):
        """Entry point for training the brain on a game environment."""
        self.control_learner.setup_environment(env)
        return self.control_learner.train(episodes, max_steps)

    def learn_from_sdr(self, sdr: Set[int]):
        """A direct learning method for pre-computed SDRs (e.g., from the VisualCortex)."""
        try:
            if self.graph.number_of_nodes() < 2:
                self._initialize_graph(sdr)
            else:
                self._stag_learning_step(sdr)
            self.iteration_count += 1
        finally:
            self._flush_events()

    def _add_node(self, sdr: Set[int], error: float = 0.1, node_type: str = 'sensory', action_name: str = None) -> int:
        node_id = (max(list(self.graph.nodes) + [0])) + 1 if self.graph.nodes else 1
        if node_type == 'action':
            node_id = -(len(self.control_learner.motor_cortex.action_nodes) + 1)

        cells = {}
        for i in range(self.network_model.cells_per_column):
            cell_id = (node_id * CELL_ID_MULTIPLIER) + i
            cells[cell_id] = {'dendrites': []}

        self.graph.add_node(node_id, prototype_sdr=sdr, error=error, cells=cells, last_active_iter=self.iteration_count,
                            type=node_type, action_name=action_name)

        if node_type != 'action':
            self.sdr_index.add(node_id, sdr)

        self.graph_has_changed = True
        self.nodes_added_since_snapshot += 1
        # if self.nodes_added_since_snapshot >= 1:
        snapshot_name = f"auto_snapshot_iter_{self.iteration_count}_nodes_{self.graph.number_of_nodes()}"
        self.create_snapshot(snapshot_name)
        self.nodes_added_since_snapshot = 0
        return node_id

    def _get_winners(self, input_sdr: Set[int], n: int = 2) -> List[Tuple[int, int]]:
        """Uses the FAISS index for fast lookups if available."""
        if self.sdr_index.index and self.sdr_index.index.ntotal > 0:
            node_ids = self.sdr_index.search(input_sdr, n)
            return [(node_id, 0) for node_id in node_ids]

        if not self.graph.nodes: return []
        sensory_nodes = [nid for nid, data in self.graph.nodes(data=True) if data.get('type') != 'action']
        if not sensory_nodes: return []

        scores = [(node_id, self.sdr_helper.overlap(input_sdr, self.graph.nodes[node_id]['prototype_sdr'])) for node_id
                  in sensory_nodes]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def _load_latest_snapshot(self):
        latest_snapshot = GraphSnapshot.objects.filter(network=self.network_model).order_by('-created_at').first()
        if latest_snapshot: self.load_from_snapshot(latest_snapshot.id)

    def load_from_snapshot(self, snapshot_id: int):
        snapshot = GraphSnapshot.objects.get(pk=snapshot_id, network=self.network_model)
        graph_data = snapshot.graph_data
        version = graph_data.get("version", "0.0")
        if version != SNAPSHOT_SCHEMA_VERSION: print(
            f"Warning: Loading snapshot with schema version {version}, but service expects {SNAPSHOT_SCHEMA_VERSION}.")
        self.graph.clear()
        for node_data in graph_data.get('nodes', []):
            node_id = int(node_data['id'])
            sdr = self.sdr_helper.list_to_sdr(node_data.get('prototype_sdr', []))
            self.graph.add_node(node_id, prototype_sdr=sdr, error=node_data.get('error', 0.0),
                                cells={int(k): v for k, v in node_data.get('cells', {}).items()},
                                last_active_iter=node_data.get('last_active_iter', 0), type=node_data.get('type'),
                                action_name=node_data.get('action_name'))
            if node_data.get('type') != 'action':
                self.sdr_index.add(node_id, sdr)
        for link_data in graph_data.get('links', []): self.graph.add_edge(int(link_data['source']),
                                                                          int(link_data['target']),
                                                                          age=link_data.get('age', 0))
        self.htm_learner.rebuild_dendrite_index()
        print(f"STAG graph loaded from snapshot '{snapshot.name}'.")

    def create_snapshot(self, name: str) -> Dict[str, Any]:
        if not name: raise ValueError("Snapshot name cannot be empty.")
        graph_data = {"version": SNAPSHOT_SCHEMA_VERSION, "nodes": [
            {'id': nid, 'prototype_sdr': self.sdr_helper.sdr_to_list(data['prototype_sdr']), 'error': data['error'],
             'cells': data['cells'], 'last_active_iter': data.get('last_active_iter', 0), 'type': data.get('type'),
             'action_name': data.get('action_name')} for nid, data in self.graph.nodes(data=True)],
                      "links": [{'source': u, 'target': v, 'age': data['age']} for u, v, data in
                                self.graph.edges(data=True)]}
        snapshot = GraphSnapshot.objects.create(network=self.network_model, name=name, graph_data=graph_data)
        all_snapshots = GraphSnapshot.objects.filter(network=self.network_model).order_by('-created_at')
        if all_snapshots.count() > 3:
            for snap in all_snapshots[3:]: snap.delete()
        return GraphSnapshotDetailSerializer(snapshot).data

    def learn_from_book(self, book_content: str):
        self.rl_component.start_reading_session()
        tokens = book_content.lower().strip().split()
        if len(tokens) < 2: return {"status": "book_too_short", "tokens": len(tokens)}
        self.learn_from_text(tokens[0])
        for i in range(len(tokens) - 1):
            current_token, actual_next_token = tokens[i], tokens[i + 1]
            prediction_result = self.predict_next_concept(current_token)
            predicted_node_id = prediction_result.get("predicted_next_node_id")
            actual_next_sdr = self.sdr_helper.encode_token(actual_next_token)
            winners = self._get_winners(actual_next_sdr, 1)
            actual_next_node_id = winners[0][0] if winners else None
            reward = 1.0 if predicted_node_id == actual_next_node_id else -0.5
            current_sdr = self.sdr_helper.encode_token(current_token)
            current_node_id = self._get_winners(current_sdr, 1)[0][0]
            if actual_next_node_id is not None:
                self.rl_component.observe_state(current_node_id, actual_next_node_id)
                self.rl_component.apply_reward(reward)
            self.learn_from_text(actual_next_token)
        self.rl_component.update_q_values()
        return {"status": "book_learned_self_supervised", "tokens": len(tokens)}

    def predict_story_continuation(self, start_text: str, max_length: int = 10) -> List[str]:
        tokens = start_text.lower().strip().split()
        if not tokens or not self.graph.nodes: return []
        input_sdr = self.sdr_helper.union([self.sdr_helper.encode_token(token) for token in tokens])
        winners = self._get_winners(input_sdr, 1)
        start_node = winners[0][0] if winners else random.choice(list(self.graph.nodes))
        path = self.rl_component.get_best_path(start_node, max_length)
        concepts = []
        for node_id in path:
            if node_id not in self.graph.nodes: continue
            node_sdr = self.graph.nodes[node_id]['prototype_sdr']
            if not self.sdr_helper.encoder_cache: continue
            closest_token = \
            max(self.sdr_helper.encoder_cache.items(), key=lambda item: self.sdr_helper.overlap(node_sdr, item[1]))[0]
            concepts.append(closest_token)
        return concepts

    def learn_from_text(self, text_content: str) -> Dict[str, Any]:
        tokens = text_content.lower().strip().split()
        if not tokens: return {"message": "No content to learn.", "status": "ignored"}
        try:
            for token in tokens:
                sdr = self.sdr_helper.encode_token(token)
                if self.graph.number_of_nodes() < 2:
                    self._initialize_graph(sdr)
                else:
                    self._stag_learning_step(sdr)
            self.iteration_count += len(tokens)
            return {"message": f"Learning step completed for {len(tokens)} tokens.", "status": "learned"}
        finally:
            self._flush_events()

    def _stag_learning_step(self, input_sdr: Set[int]):
        if self.graph.number_of_nodes() < 2: return
        winners = self._get_winners(input_sdr, 2)
        if len(winners) < 2: return
        s1_id, s2_id = winners[0][0], winners[1][0]
        self.graph.nodes[s1_id]['last_active_iter'] = self.iteration_count
        active_columns = {s1_id}
        active_cells_this_step = self.htm_learner.activate_cells(active_columns)
        self.htm_learner.learn_on_dendrites(active_cells_this_step)
        self.predictive_cells_t = self.htm_learner.calculate_predictions(active_cells_this_step)
        self.active_cells_t_minus_1 = active_cells_this_step
        self.gng_learner.update_spatial_topology(s1_id, s2_id, input_sdr)

    def _initialize_graph(self, initial_sdr: Set[int]):
        if self.graph.number_of_nodes() == 0:
            self._add_node(sdr=initial_sdr)
        elif self.graph.number_of_nodes() == 1:
            second_sdr = self.sdr_helper.create_random_sdr()
            node1_id = next(iter(self.graph.nodes))
            node2_id = self._add_node(sdr=second_sdr)
            self._add_edge(node1_id, node2_id)

    def _remove_node(self, node_id: int):
        if self.graph.has_node(node_id):
            cells_to_remove = self.graph.nodes[node_id].get('cells', {}).keys()
            for cell_id in cells_to_remove: self.htm_learner.dendrite_index.pop(cell_id, None)
            self.graph.remove_node(node_id)
            self.graph_has_changed = True

    def _add_edge(self, u: int, v: int, age: int = 0):
        if u == v: return
        if self.graph.has_edge(u, v):
            self.graph.edges[u, v]['age'] = 0
        else:
            self.graph.add_edge(u, v, age=age); self.graph_has_changed = True

    def _remove_edge(self, u: int, v: int):
        if self.graph.has_edge(u, v): self.graph.remove_edge(u, v); self.graph_has_changed = True

    def _flush_events(self):
        if self.channel_layer and self.graph_has_changed:
            print("Graph structure changed, broadcasting new state...")
            graph_state = self.get_graph_state()
            message = {'type': 'graph_state_update', 'payload': graph_state}
            async_to_sync(self.channel_layer.group_send)(self.channel_group_name, message)
            self.graph_has_changed = False

    def _get_node_id_for_cell(self, cell_id: int) -> int:
        return cell_id // CELL_ID_MULTIPLIER

    def predict_next_concept(self, text_content: str) -> Dict[str, Any]:
        tokens = text_content.lower().strip().split()
        if not self.graph.nodes or not tokens: return {"error": "The brain is empty or the query is empty."}
        local_active_cells, local_predictive_cells = set(), self.predictive_cells_t.copy()
        for token in tokens:
            input_sdr = self.sdr_helper.encode_token(token)
            winners = self._get_winners(input_sdr, 1)
            if not winners: continue
            winner_id, _ = winners[0]
            column_node = self.graph.nodes[winner_id]
            cells_in_column = set(column_node['cells'].keys())
            predicted_cells_in_col = cells_in_column.intersection(local_predictive_cells)
            local_active_cells = predicted_cells_in_col if predicted_cells_in_col else cells_in_column
            local_predictive_cells = self.htm_learner.calculate_predictions(local_active_cells)
        if not local_predictive_cells: return {"prediction": "No specific prediction could be made."}
        predicted_column_ids = [self._get_node_id_for_cell(cell_id) for cell_id in local_predictive_cells]
        most_common_column = Counter(predicted_column_ids).most_common(1)[0]
        predicted_node_id, prediction_strength = most_common_column[0], most_common_column[1]
        predicted_node_data = self.graph.nodes[predicted_node_id]
        result = {"predicted_next_node_id": predicted_node_id, "prediction_strength": prediction_strength,
                  "representative_sdr": self.sdr_helper.sdr_to_list(predicted_node_data['prototype_sdr'])}
        return result

    def get_graph_state(self) -> Dict[str, List]:
        nodes_serializable = []
        for node_id, data in self.graph.nodes(data=True):
            node_copy = data.copy()
            node_copy['id'] = node_id
            node_copy['prototype_sdr'] = self.sdr_helper.sdr_to_list(node_copy['prototype_sdr'])
            del node_copy['cells']
            nodes_serializable.append(node_copy)
        return {"nodes": nodes_serializable,
                "links": [{"source": u, "target": v, **data} for u, v, data in self.graph.edges(data=True)]}

    def run_tests(self):
        print("\n--- RUNNING STAG NETWORK TESTS ---")
        test_service = STAGNetworkService(self.network_model.id)
        test_service.graph.clear()
        test_service.htm_learner.rebuild_dendrite_index()
        print("\n[Test 1: Edge Cases]")
        assert self.predict_story_continuation("") == [], "Test failed: Empty input should return empty list."
        print("  - Passed: Empty input.")
        self.learn_from_text("apple")
        assert self.graph.number_of_nodes() == 1, "Test failed: First token should create one node."
        print("  - Passed: Single token learning.")
        print("\n[Test 2: Self-Supervised RL Prediction]")
        test_service.graph.clear()
        test_service.htm_learner.rebuild_dendrite_index()
        book_content = "cat chases mouse mouse eats cheese"
        test_service.learn_from_book(book_content)
        prediction_prompt = "cat"
        result_concepts = self.predict_story_continuation(prediction_prompt)
        assert "chases" in result_concepts, f"Test failed: Story continuation did not include 'chases'. Got: {result_concepts}"
        assert "mouse" in result_concepts, f"Test failed: Story continuation did not include 'mouse'. Got: {result_concepts}"
        print(f"  - Passed: Correctly predicted story continuation: {' '.join(result_concepts)}")
        print("\n--- ALL TESTS PASSED ---\n")
        return {"status": "All tests passed"}