import numpy as np
import random
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EnvironmentState:
    """Represents the current state of an environment"""
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Configuration for a game environment."""
    name: str
    size: int = 10
    difficulty: str = "medium"
    # Add other common configuration parameters if needed by services.py
    # For example, specific parameters for GridWorld, Maze, Snake
    # These might be passed as kwargs to the environment constructor
    # Example:
    # gridworld_obstacles: int = 0
    # maze_complexity: str = "medium"
    # snake_initial_length: int = 1


class BaseEnvironment(ABC):
    """Base class for all game environments"""

    def __init__(self):
        self.action_space = []
        self.observation_space = None
        self.current_state = None
        self.done = False

    @abstractmethod
    def reset(self) -> EnvironmentState:
        """Reset environment to initial state"""
        pass

    @abstractmethod
    def step(self, action: str) -> EnvironmentState:
        """Take action and return (observation, reward, done, info)"""
        pass

    @abstractmethod
    def render(self) -> str:
        """Return string representation of current state"""
        pass

    def get_state_size(self) -> int:
        """Return the size of the state space"""
        raise NotImplementedError

    def get_action_size(self) -> int:
        """Return the size of the action space"""
        return len(self.action_space)


class GridWorldEnvironment(BaseEnvironment):
    """Simple grid world with obstacles and goal"""

    def __init__(self, size: int = 8):
        super().__init__()
        self.size = size
        self.action_space = ['up', 'down', 'left', 'right']
        self.grid = np.zeros((size, size))
        self.player_pos = [0, 0]
        self.goal_pos = [size - 1, size - 1]
        self.obstacles = []
        self.done = False
        self.steps_taken = 0
        self.max_steps = size * size * 2  # Reasonable step limit
        self._generate_obstacles()

    def _generate_obstacles(self):
        """Generate random obstacles"""
        num_obstacles = self.size // 2
        for _ in range(num_obstacles):
            attempts = 0
            while attempts < 50:  # Prevent infinite loop
                pos = [random.randint(1, self.size - 2), random.randint(1, self.size - 2)]
                if pos != self.goal_pos and pos != self.player_pos and pos not in self.obstacles:
                    self.obstacles.append(pos)
                    break
                attempts += 1

    def reset(self) -> EnvironmentState:
        """Reset to starting position"""
        self.player_pos = [0, 0]
        self.done = False
        self.steps_taken = 0
        observation = self._get_observation()
        return EnvironmentState(
            observation=observation,
            reward=0.0,
            done=False,
            info={"steps": self.steps_taken}
        )

    def _get_observation(self) -> np.ndarray:
        """Get current state as observation"""
        obs = np.zeros((self.size, self.size))

        # Mark obstacles
        for obs_pos in self.obstacles:
            obs[obs_pos[0], obs_pos[1]] = -1

        # Mark goal
        obs[self.goal_pos[0], self.goal_pos[1]] = 1

        # Mark player
        obs[self.player_pos[0], self.player_pos[1]] = 0.5

        return obs.flatten()  # Flatten for consistent observation space

    def get_state_size(self) -> int:
        return self.size * self.size

    def step(self, action: str) -> EnvironmentState:
        """Take action and return result"""
        if self.done:
            # Environment is already done, return current state
            return EnvironmentState(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={"steps": self.steps_taken, "reason": "already_done"}
            )

        old_pos = self.player_pos.copy()
        self.steps_taken += 1

        # Move based on action
        if action == 'up' and self.player_pos[0] > 0:
            self.player_pos[0] -= 1
        elif action == 'down' and self.player_pos[0] < self.size - 1:
            self.player_pos[0] += 1
        elif action == 'left' and self.player_pos[1] > 0:
            self.player_pos[1] -= 1
        elif action == 'right' and self.player_pos[1] < self.size - 1:
            self.player_pos[1] += 1

        reward = 0.0
        info = {"steps": self.steps_taken}

        # Check for collision with obstacles
        if self.player_pos in self.obstacles:
            self.player_pos = old_pos  # Revert move
            reward = -0.1
            info["collision"] = True

        # Check for goal
        elif self.player_pos == self.goal_pos:
            reward = 10.0
            self.done = True
            info["success"] = True

        # Check for step limit
        elif self.steps_taken >= self.max_steps:
            reward = -1.0
            self.done = True
            info["timeout"] = True

        else:
            # Small negative reward for each step to encourage efficiency
            reward = -0.01

        return EnvironmentState(
            observation=self._get_observation(),
            reward=reward,
            done=self.done,
            info=info
        )

    def render(self) -> str:
        """Render current state"""
        display = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if [i, j] == self.player_pos:
                    row.append('P')
                elif [i, j] == self.goal_pos:
                    row.append('G')
                elif [i, j] in self.obstacles:
                    row.append('X')
                else:
                    row.append('.')
            display.append(' '.join(row))
        return '\n'.join(display)


class MazeEnvironment(BaseEnvironment):
    """Maze environment with procedural generation"""

    def __init__(self, size: int = 10):
        super().__init__()
        self.size = size if size % 2 == 1 else size + 1  # Ensure odd size for maze generation
        self.action_space = ['up', 'down', 'left', 'right']
        self.maze = None
        self.player_pos = [1, 1]
        self.goal_pos = [self.size - 2, self.size - 2]
        self.done = False
        self.steps_taken = 0
        self.max_steps = self.size * self.size
        self._generate_maze()

    def _generate_maze(self):
        """Generate a solvable maze using recursive backtracking"""
        # Initialize maze with walls
        self.maze = np.ones((self.size, self.size))

        # Create paths
        def carve_path(x, y):
            self.maze[x, y] = 0  # Mark as path

            # Randomize directions
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (1 <= nx < self.size - 1 and 1 <= ny < self.size - 1 and
                        self.maze[nx, ny] == 1):
                    # Carve wall between current and next cell
                    self.maze[x + dx // 2, y + dy // 2] = 0
                    carve_path(nx, ny)

        # Start carving from (1,1)
        carve_path(1, 1)

        # Ensure goal is reachable
        self.maze[self.goal_pos[0], self.goal_pos[1]] = 0

        # Add some extra paths for complexity
        for _ in range(self.size // 4):
            x, y = random.randint(1, self.size - 2), random.randint(1, self.size - 2)
            self.maze[x, y] = 0

    def reset(self) -> EnvironmentState:
        """Reset to starting position"""
        self.player_pos = [1, 1]
        self.done = False
        self.steps_taken = 0
        observation = self._get_observation()
        return EnvironmentState(
            observation=observation,
            reward=0.0,
            done=False,
            info={"steps": self.steps_taken}
        )

    def _get_observation(self) -> np.ndarray:
        """Get current state as observation"""
        obs = self.maze.copy()

        # Mark player position
        obs[self.player_pos[0], self.player_pos[1]] = 0.5

        # Mark goal position
        obs[self.goal_pos[0], self.goal_pos[1]] = -0.5

        return obs.flatten()  # Flatten for consistent observation space

    def get_state_size(self) -> int:
        return self.size * self.size

    def step(self, action: str) -> EnvironmentState:
        """Take action and return result"""
        if self.done:
            return EnvironmentState(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={"steps": self.steps_taken, "reason": "already_done"}
            )

        old_pos = self.player_pos.copy()
        self.steps_taken += 1

        # Move based on action
        if action == 'up':
            self.player_pos[0] -= 1
        elif action == 'down':
            self.player_pos[0] += 1
        elif action == 'left':
            self.player_pos[1] -= 1
        elif action == 'right':
            self.player_pos[1] += 1

        reward = 0.0
        info = {"steps": self.steps_taken}

        # Check bounds and walls
        if (self.player_pos[0] < 0 or self.player_pos[0] >= self.size or
                self.player_pos[1] < 0 or self.player_pos[1] >= self.size or
                self.maze[self.player_pos[0], self.player_pos[1]] == 1):
            self.player_pos = old_pos  # Revert move
            reward = -0.1
            info["collision"] = True

        # Check for goal
        elif self.player_pos == self.goal_pos:
            reward = 10.0
            self.done = True
            info["success"] = True

        # Check for step limit
        elif self.steps_taken >= self.max_steps:
            reward = -1.0
            self.done = True
            info["timeout"] = True

        else:
            # Small negative reward for each step
            reward = -0.01

        return EnvironmentState(
            observation=self._get_observation(),
            reward=reward,
            done=self.done,
            info=info
        )

    def render(self) -> str:
        """Render current state"""
        display = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if [i, j] == self.player_pos:
                    row.append('P')
                elif [i, j] == self.goal_pos:
                    row.append('G')
                elif self.maze[i, j] == 1:
                    row.append('█')
                else:
                    row.append(' ')
            display.append(''.join(row))
        return '\n'.join(display)


class SnakeEnvironment(BaseEnvironment):
    """Classic Snake game"""

    def __init__(self, size: int = 10):
        super().__init__()
        self.size = size
        self.action_space = ['up', 'down', 'left', 'right']
        self.snake = [[size // 2, size // 2]]
        self.direction = 'right'
        self.food_pos = None
        self.score = 0
        self.done = False
        self.steps_taken = 0
        self.max_steps = size * size * 4  # Generous step limit
        self._place_food()

    def _place_food(self):
        """Place food at random location not occupied by snake"""
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
            if pos not in self.snake:
                self.food_pos = pos
                break
            attempts += 1

    def reset(self) -> EnvironmentState:
        """Reset game state"""
        self.snake = [[self.size // 2, self.size // 2]]
        self.direction = 'right'
        self.score = 0
        self.done = False
        self.steps_taken = 0
        self._place_food()
        observation = self._get_observation()
        return EnvironmentState(
            observation=observation,
            reward=0.0,
            done=False,
            info={"steps": self.steps_taken, "score": self.score}
        )

    def _get_observation(self) -> np.ndarray:
        """Get current state as observation"""
        obs = np.zeros((self.size, self.size))

        # Mark snake body
        for segment in self.snake:
            obs[segment[0], segment[1]] = 0.5

        # Mark snake head
        if self.snake:
            head = self.snake[0]
            obs[head[0], head[1]] = 1.0

        # Mark food
        if self.food_pos:
            obs[self.food_pos[0], self.food_pos[1]] = -1.0

        return obs.flatten()  # Flatten for consistent observation space

    def get_state_size(self) -> int:
        return self.size * self.size

    def step(self, action: str) -> EnvironmentState:
        """Take action and return result"""
        if self.done:
            return EnvironmentState(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={"steps": self.steps_taken, "score": self.score, "reason": "already_done"}
            )

        self.steps_taken += 1

        # Update direction (can't reverse)
        if action == 'up' and self.direction != 'down':
            self.direction = 'up'
        elif action == 'down' and self.direction != 'up':
            self.direction = 'down'
        elif action == 'left' and self.direction != 'right':
            self.direction = 'left'
        elif action == 'right' and self.direction != 'left':
            self.direction = 'right'

        # Move snake head
        head = self.snake[0].copy()
        if self.direction == 'up':
            head[0] -= 1
        elif self.direction == 'down':
            head[0] += 1
        elif self.direction == 'left':
            head[1] -= 1
        elif self.direction == 'right':
            head[1] += 1

        reward = 0.0
        info = {"steps": self.steps_taken, "score": self.score}

        # Check wall collision
        if (head[0] < 0 or head[0] >= self.size or
                head[1] < 0 or head[1] >= self.size):
            reward = -10.0
            self.done = True
            info["wall_collision"] = True

        # Check self collision
        elif head in self.snake:
            reward = -10.0
            self.done = True
            info["self_collision"] = True

        else:
            # Add new head
            self.snake.insert(0, head)

            # Check food collision
            if head == self.food_pos:
                self.score += 1
                self._place_food()
                reward = 10.0
                info["food_eaten"] = True
            else:
                # Remove tail if no food eaten
                self.snake.pop()
                reward = 0.1  # Small positive reward for staying alive

            # Check step limit
            if self.steps_taken >= self.max_steps:
                reward = -1.0
                self.done = True
                info["timeout"] = True

        return EnvironmentState(
            observation=self._get_observation(),
            reward=reward,
            done=self.done,
            info=info
        )

    def render(self) -> str:
        """Render current state"""
        display = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if [i, j] == self.snake[0]:  # Head
                    row.append('H')
                elif [i, j] in self.snake:  # Body
                    row.append('S')
                elif [i, j] == self.food_pos:  # Food
                    row.append('F')
                else:
                    row.append('.')
            display.append(' '.join(row))
        return '\n'.join(display) + f'\nScore: {self.score}'


class EnvironmentManager:
    """Manager for creating and listing environments"""

    def __init__(self):
        self.environments = {
            'gridworld': GridWorldEnvironment,
            'maze': MazeEnvironment,
            'snake': SnakeEnvironment
        }

    def create_environment(self, env_name: str, **kwargs) -> BaseEnvironment:
        """Create environment instance"""
        if env_name not in self.environments:
            raise ValueError(f"Unknown environment: {env_name}")

        return self.environments[env_name](**kwargs)

    def get_available_environments(self) -> List[str]:
        """Get list of available environment names"""
        return list(self.environments.keys())

    def get_environment_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all environments"""
        return {
            'gridworld': 'Simple grid navigation with obstacles - reach the goal while avoiding obstacles',
            'maze': 'Navigate through a randomly generated maze to reach the exit',
            'snake': 'Classic snake game - eat food and grow without hitting walls or yourself'
        }


# Global instances and functions for backward compatibility
ENVIRONMENTS = {
    'gridworld': GridWorldEnvironment,
    'maze': MazeEnvironment,
    'snake': SnakeEnvironment
}


def create_environment(env_name: str, **kwargs) -> BaseEnvironment:
    """Create environment instance"""
    manager = EnvironmentManager()
    return manager.create_environment(env_name, **kwargs)


def get_environment_descriptions() -> Dict[str, str]:
    """Get environment descriptions"""
    manager = EnvironmentManager()
    return manager.get_environment_descriptions()
