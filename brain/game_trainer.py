import numpy as np
import random
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import threading
import json

from .environments import EnvironmentManager, BaseEnvironment, EnvironmentState
from .models import BrainNetwork


class ControlLearner:
    """Q-Learning agent for game control"""

    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Q-table for discrete states
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.experience_buffer = deque(maxlen=10000)

    def _discretize_state(self, state: np.ndarray) -> str:
        """Convert continuous state to discrete string for Q-table"""
        # Simple discretization - round to 2 decimal places
        discrete_state = np.round(state, 2)
        return str(discrete_state.tolist())

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        state_key = self._discretize_state(state)

        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state_key])

    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        """Update Q-table using Q-learning"""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)

        # Q-learning update
        current_q = self.q_table[state_key][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])

        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Save experience for potential replay"""
        self.experience_buffer.append((state, action, reward, next_state, done))


class GameTrainer:
    """Main trainer class that integrates with STAG network"""

    def __init__(self, network_id: int):
        self.network_id = network_id
        self.network_model = None
        self.stag_service = None
        self.env_manager = EnvironmentManager()
        self.control_learner = None
        self.visualization_enabled = True
        self.training_stats = {
            'episodes_completed': 0,
            'total_reward': 0,
            'success_count': 0,
            'episode_rewards': [],
            'episode_successes': []
        }

        # Initialize network connection
        self._initialize_network()

    def _initialize_network(self):
        """Initialize connection to STAG network"""
        try:
            self.network_model = BrainNetwork.objects.get(id=self.network_id)

            # Try to get STAG service if available
            try:
                from .services import STAGNetworkService
                self.stag_service = STAGNetworkService(self.network_id)
                print(f"Connected to STAG network {self.network_id}")
            except Exception as e:
                print(f"STAG service not available, using fallback Q-learning: {e}")
                self.stag_service = None

        except BrainNetwork.DoesNotExist:
            print(f"Network {self.network_id} not found, using standalone mode")
            self.network_model = None
            self.stag_service = None

    def train_single_episode(self, environment_name: str) -> Dict:
        """Train for a single episode"""
        env = self.env_manager.create_environment(environment_name)

        # Initialize control learner if not exists
        if self.control_learner is None:
            self.control_learner = ControlLearner(
                env.get_state_size(),
                env.get_action_size()
            )

        # Reset environment and get initial state
        env_state = env.reset()
        state = env_state.observation
        total_reward = 0
        steps = 0
        max_steps = 1000

        while not env_state.done and steps < max_steps:
            # Choose action
            if self.stag_service:
                # Use STAG network for action selection
                action_idx = self._get_stag_action(state, env)
            else:
                # Use Q-learning fallback
                action_idx = self.control_learner.choose_action(state, training=True)

            # Convert action index to action name
            action = env.action_space[action_idx]

            # Take action
            env_state = env.step(action)
            next_state = env_state.observation
            reward = env_state.reward
            done = env_state.done

            total_reward += reward

            # Learn from experience
            if self.stag_service:
                self._update_stag_network(state, action_idx, reward, next_state, done)
            else:
                self.control_learner.learn(state, action_idx, reward, next_state, done)

            state = next_state
            steps += 1

            if self.visualization_enabled and steps % 10 == 0:
                print(env.render())
                time.sleep(0.1)

        # Update statistics
        success = total_reward > 0
        self.training_stats['episodes_completed'] += 1
        self.training_stats['total_reward'] += total_reward
        self.training_stats['episode_rewards'].append(total_reward)
        self.training_stats['episode_successes'].append(success)
        if success:
            self.training_stats['success_count'] += 1

        return {
            'episode': self.training_stats['episodes_completed'],
            'total_reward': total_reward,
            'steps': steps,
            'success': success,
            'info': env_state.info
        }

    def train_multiple_episodes(self, episodes: int, environment_name: str):
        """Train for multiple episodes"""
        print(f"Starting training on {environment_name} for {episodes} episodes...")

        for episode in range(episodes):
            result = self.train_single_episode(environment_name)

            if episode % 10 == 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-10:])
                success_rate = np.mean(self.training_stats['episode_successes'][-10:])
                print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")

        print(f"Training completed! Total episodes: {episodes}")

    def evaluate_performance(self, num_episodes: int, environment_name: str) -> Dict:
        """Evaluate the trained agent"""
        print(f"Evaluating performance on {environment_name} for {num_episodes} episodes...")

        rewards = []
        successes = []

        # Temporarily disable exploration for evaluation
        original_epsilon = getattr(self.control_learner, 'epsilon', 0) if self.control_learner else 0
        if self.control_learner:
            self.control_learner.epsilon = 0

        for episode in range(num_episodes):
            env = self.env_manager.create_environment(environment_name)
            env_state = env.reset()
            state = env_state.observation
            total_reward = 0
            steps = 0
            max_steps = 1000

            while not env_state.done and steps < max_steps:
                if self.stag_service:
                    action_idx = self._get_stag_action(state, env)
                else:
                    action_idx = self.control_learner.choose_action(state, training=False)

                action = env.action_space[action_idx]
                env_state = env.step(action)
                state = env_state.observation
                total_reward += env_state.reward
                steps += 1

            rewards.append(total_reward)
            successes.append(total_reward > 0)

        # Restore original epsilon
        if self.control_learner:
            self.control_learner.epsilon = original_epsilon

        results = {
            'avg_reward': np.mean(rewards),
            'success_rate': np.mean(successes),
            'rewards': rewards,
            'successes': successes
        }

        print(f"Evaluation complete: Avg Reward: {results['avg_reward']:.2f}, "
              f"Success Rate: {results['success_rate']:.2f}")

        return results

    def run_curriculum(self):
        """Run curriculum learning across multiple environments"""
        curriculum = [
            ('gridworld', 30),
            ('maze', 50),
            ('snake', 75)
        ]

        print("Starting curriculum training...")
        for env_name, episodes in curriculum:
            print(f"\n--- Training on {env_name} for {episodes} episodes ---")
            self.train_multiple_episodes(episodes, env_name)

            # Evaluate after each stage
            results = self.evaluate_performance(10, env_name)
            print(f"Curriculum stage complete - {env_name}: {results['success_rate']:.2f} success rate")

        print("\nCurriculum training completed!")

    def _get_stag_action(self, state: np.ndarray, env: BaseEnvironment) -> int:
        """Get action from STAG network"""
        try:
            # Convert state to text representation for STAG
            state_text = f"game_state_{hash(str(state)) % 10000}"

            # Use STAG for prediction
            prediction = self.stag_service.predict_next_concept(state_text)

            # Convert prediction to action (simple mapping)
            action_hash = hash(str(prediction)) % env.get_action_size()
            return action_hash

        except Exception as e:
            print(f"STAG action selection failed: {e}")
            # Fallback to random action
            return random.randint(0, env.get_action_size() - 1)

    def _update_stag_network(self, state: np.ndarray, action: int, reward: float,
                             next_state: np.ndarray, done: bool):
        """Update STAG network with game experience"""
        try:
            # Convert experience to text for STAG learning
            experience_text = f"state_{hash(str(state)) % 1000}_action_{action}_reward_{reward:.1f}"

            # Learn from experience
            self.stag_service.learn_from_text(experience_text)

        except Exception as e:
            print(f"STAG network update failed: {e}")

    def get_training_statistics(self) -> Dict:
        """Get current training statistics"""
        if not self.training_stats['episode_rewards']:
            return {
                'episodes_completed': 0,
                'avg_reward': 0,
                'success_rate': 0,
                'total_reward': 0
            }

        return {
            'episodes_completed': self.training_stats['episodes_completed'],
            'avg_reward': np.mean(self.training_stats['episode_rewards']),
            'success_rate': np.mean(self.training_stats['episode_successes']),
            'total_reward': self.training_stats['total_reward'],
            'recent_rewards': self.training_stats['episode_rewards'][-10:],
            'recent_successes': self.training_stats['episode_successes'][-10:]
        }


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='STAG Game Trainer')
    parser.add_argument('--network-id', type=int, required=True, help='Brain network ID')
    parser.add_argument('--environment', type=str, default='maze',
                        choices=['gridworld', 'maze', 'snake'], help='Game environment')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--evaluate', type=int, default=0, help='Number of evaluation episodes')
    parser.add_argument('--curriculum', action='store_true', help='Run curriculum training')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    trainer = GameTrainer(args.network_id)

    if args.interactive:
        print("Interactive STAG Game Trainer")
        print("Commands: train <env> <episodes>, eval <env> <episodes>, curriculum, stats, quit")

        while True:
            try:
                command = input("STAG> ").strip().split()
                if not command:
                    continue

                if command[0] == 'quit':
                    break
                elif command[0] == 'train' and len(command) >= 3:
                    env_name, episodes = command[1], int(command[2])
                    trainer.train_multiple_episodes(episodes, env_name)
                elif command[0] == 'eval' and len(command) >= 3:
                    env_name, episodes = command[1], int(command[2])
                    results = trainer.evaluate_performance(episodes, env_name)
                    print(f"Results: {results}")
                elif command[0] == 'curriculum':
                    trainer.run_curriculum()
                elif command[0] == 'stats':
                    stats = trainer.get_training_statistics()
                    print(f"Training Statistics: {stats}")
                else:
                    print("Unknown command")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

    elif args.curriculum:
        trainer.run_curriculum()
    else:
        trainer.train_multiple_episodes(args.episodes, args.environment)

        if args.evaluate > 0:
            results = trainer.evaluate_performance(args.evaluate, args.environment)
            print(f"Final evaluation results: {results}")
