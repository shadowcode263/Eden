import numpy as np
import random
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class EntityType(Enum):
    """Defines the numerical representation for entities in the grid."""
    EMPTY = 0.0
    PLAYER = 0.5
    GOAL = 1.0
    OBSTACLE = -1.0
    SNAKE_BODY = 0.5
    SNAKE_HEAD = 1.0
    FOOD = -1.0


@dataclass
class EnvironmentState:
    """Represents the current state of an environment."""
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    """Abstract base class for all game environments."""

    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.action_space: List[str] = []
        self.done = False

    @abstractmethod
    def reset(self) -> EnvironmentState:
        """Resets the environment to its initial state."""
        pass

    @abstractmethod
    def step(self, action: str) -> EnvironmentState:
        """Takes an action and returns the new state, reward, and done status."""
        pass

    @abstractmethod
    def render(self) -> str:
        """Returns a string representation of the current state for display."""
        pass

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the observation space."""
        return (self.size, self.size)

    @property
    def action_size(self) -> int:
        """Returns the number of possible actions."""
        return len(self.action_space)


class GridWorldEnvironment(BaseEnvironment):
    """A simple grid world where an agent must navigate to a goal while avoiding obstacles."""

    def __init__(self, name: str, size: int = 8):
        super().__init__(name, size)
        self.action_space = ['up', 'down', 'left', 'right']
        self.player_pos: List[int] = []
        self.goal_pos: List[int] = []
        self.obstacles: List[List[int]] = []
        self.steps_taken = 0
        self.max_steps = self.size * self.size

    def _generate_layout(self):
        """Generates the player, goal, and obstacle positions."""
        self.player_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.obstacles = []
        num_obstacles = self.size // 2
        for _ in range(num_obstacles):
            while True:
                pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
                if pos != self.player_pos and pos != self.goal_pos and pos not in self.obstacles:
                    self.obstacles.append(pos)
                    break

    def reset(self) -> EnvironmentState:
        self._generate_layout()
        self.done = False
        self.steps_taken = 0
        return EnvironmentState(observation=self._get_observation(), reward=0.0, done=self.done)

    def _get_observation(self) -> np.ndarray:
        obs = np.full((self.size, self.size), EntityType.EMPTY.value)
        for ob_pos in self.obstacles:
            obs[ob_pos[0], ob_pos[1]] = EntityType.OBSTACLE.value
        obs[self.goal_pos[0], self.goal_pos[1]] = EntityType.GOAL.value
        obs[self.player_pos[0], self.player_pos[1]] = EntityType.PLAYER.value
        return obs

    def step(self, action: str) -> EnvironmentState:
        if self.done:
            return EnvironmentState(self._get_observation(), 0.0, True, {"reason": "already_done"})

        self.steps_taken += 1
        old_pos = self.player_pos.copy()
        old_dist = abs(old_pos[0] - self.goal_pos[0]) + abs(old_pos[1] - self.goal_pos[1])

        if action == 'up' and self.player_pos[0] > 0:
            self.player_pos[0] -= 1
        elif action == 'down' and self.player_pos[0] < self.size - 1:
            self.player_pos[0] += 1
        elif action == 'left' and self.player_pos[1] > 0:
            self.player_pos[1] -= 1
        elif action == 'right' and self.player_pos[1] < self.size - 1:
            self.player_pos[1] += 1

        new_dist = abs(self.player_pos[0] - self.goal_pos[0]) + abs(self.player_pos[1] - self.goal_pos[1])

        info = {"steps": self.steps_taken}
        if self.player_pos in self.obstacles:
            self.player_pos = old_pos
            reward = -10.0
            info["collision"] = True
        elif self.player_pos == self.goal_pos:
            reward = 50.0
            self.done = True
            info["success"] = True
        elif self.steps_taken >= self.max_steps:
            reward = -5.0
            self.done = True
            info["timeout"] = True
        else:
            # Reward for getting closer, penalize for moving away or standing still
            distance_reward = (old_dist - new_dist)
            step_penalty = -0.1
            reward = distance_reward + step_penalty

        return EnvironmentState(self._get_observation(), reward, self.done, info)

    def render(self) -> str:
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        for x, y in self.obstacles: grid[x][y] = 'X'
        grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
        grid[self.player_pos[0], self.player_pos[1]] = 'P'
        return "\n".join(" ".join(row) for row in grid)


class MazeEnvironment(GridWorldEnvironment):
    """A maze environment that inherits from GridWorld and overrides layout generation."""

    def __init__(self, name: str, size: int = 11):
        # Maze size must be odd
        super().__init__(name, size if size % 2 == 1 else size + 1)

    def _generate_layout(self):
        """Generates a solvable maze using recursive backtracking."""
        self.player_pos = [1, 1]
        self.goal_pos = [self.size - 2, self.size - 2]

        # In this context, obstacles are walls
        maze = np.full((self.size, self.size), EntityType.OBSTACLE.value)

        def carve(x, y):
            maze[x, y] = EntityType.EMPTY.value
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and maze[nx, ny] == EntityType.OBSTACLE.value:
                    maze[x + dx // 2, y + dy // 2] = EntityType.EMPTY.value
                    carve(nx, ny)

        carve(self.player_pos[0], self.player_pos[1])
        # Ensure the goal is reachable
        maze[self.goal_pos[0], self.goal_pos[1]] = EntityType.EMPTY.value

        self.obstacles = list(zip(*np.where(maze == EntityType.OBSTACLE.value)))


class SnakeEnvironment(BaseEnvironment):
    """Classic Snake game environment."""

    def __init__(self, name: str, size: int = 10):
        super().__init__(name, size)
        self.action_space = ['up', 'down', 'left', 'right']
        self.snake: List[List[int]] = []
        self.direction = 'right'
        self.food_pos: List[int] = []
        self.score = 0
        self.steps_since_food = 0
        self.max_steps_without_food = self.size * self.size

    def _place_food(self):
        while True:
            pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
            if pos not in self.snake:
                self.food_pos = pos
                return

    def reset(self) -> EnvironmentState:
        self.snake = [[self.size // 2, self.size // 2]]
        self.direction = 'right'
        self.score = 0
        self.done = False
        self.steps_since_food = 0
        self._place_food()
        return EnvironmentState(self._get_observation(), 0.0, self.done)

    def _get_observation(self) -> np.ndarray:
        obs = np.full((self.size, self.size), EntityType.EMPTY.value)
        obs[self.food_pos[0], self.food_pos[1]] = EntityType.FOOD.value
        for segment in self.snake:
            obs[segment[0], segment[1]] = EntityType.SNAKE_BODY.value
        # Head is distinct
        obs[self.snake[0][0], self.snake[0][1]] = EntityType.SNAKE_HEAD.value
        return obs

    def step(self, action: str) -> EnvironmentState:
        if self.done:
            return EnvironmentState(self._get_observation(), 0.0, True, {"reason": "already_done"})

        self.steps_since_food += 1

        # Prevent snake from reversing on itself
        if (action == 'up' and self.direction != 'down') or \
                (action == 'down' and self.direction != 'up') or \
                (action == 'left' and self.direction != 'right') or \
                (action == 'right' and self.direction != 'left'):
            self.direction = action

        head = self.snake[0].copy()
        if self.direction == 'up':
            head[0] -= 1
        elif self.direction == 'down':
            head[0] += 1
        elif self.direction == 'left':
            head[1] -= 1
        elif self.direction == 'right':
            head[1] += 1

        info = {"score": self.score}
        if not (0 <= head[0] < self.size and 0 <= head[1] < self.size) or head in self.snake:
            reward = -100.0  # Harsh penalty for dying
            self.done = True
            info["death"] = "wall_collision" if not (
                        0 <= head[0] < self.size and 0 <= head[1] < self.size) else "self_collision"
        else:
            self.snake.insert(0, head)
            if head == self.food_pos:
                self.score += 1
                self.steps_since_food = 0
                self._place_food()
                reward = 100.0  # Large reward for eating
                info["food_eaten"] = True
            else:
                self.snake.pop()
                # Reward for getting closer to food
                old_dist = abs(self.snake[1][0] - self.food_pos[0]) + abs(self.snake[1][1] - self.food_pos[1])
                new_dist = abs(head[0] - self.food_pos[0]) + abs(head[1] - self.food_pos[1])
                reward = (old_dist - new_dist) * 0.5 - 0.1  # Small step penalty

        if self.steps_since_food > self.max_steps_without_food:
            self.done = True
            info["death"] = "starvation"

        return EnvironmentState(self._get_observation(), reward, self.done, info)

    def render(self) -> str:
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        grid[self.food_pos[0]][self.food_pos[1]] = 'F'
        for x, y in self.snake: grid[x][y] = 'S'
        grid[self.snake[0][0]][self.snake[0][1]] = 'H'
        return "\n".join(" ".join(row) for row in grid) + f"\nScore: {self.score}"


class EnvironmentManager:
    """A centralized manager for creating and listing game environments."""

    _environments = {
        'gridworld': GridWorldEnvironment,
        'maze': MazeEnvironment,
        'snake': SnakeEnvironment
    }

    @classmethod
    def get_env(cls, env_name: str, **kwargs) -> BaseEnvironment:
        """Creates an instance of a specified environment."""
        if env_name not in cls._environments:
            raise ValueError(f"Unknown environment: '{env_name}'. Available: {list(cls._environments.keys())}")
        return cls._environments[env_name](name=env_name, **kwargs)

    @classmethod
    def get_environment_descriptions(cls) -> Dict[str, str]:
        """Returns a dictionary of available environments and their descriptions."""
        return {
            'gridworld': 'A simple grid navigation task. The agent must reach the goal (G) while avoiding obstacles (X).',
            'maze': 'A more complex navigation task where the agent must find its way through a procedurally generated maze.',
            'snake': 'The classic arcade game. The agent (H) must eat food (F) to grow longer (S) without hitting walls or itself.'
        }
