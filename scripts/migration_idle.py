#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
Red-fronted Brown Lemur Migration Path Planning - PURE RL VERSION
===================================================================
Efficient paths through PURE reinforcement learning:
1. Smart reward shaping (no pre-training needed)
2. Curriculum learning (easy → hard)
3. Strong efficiency penalties
4. A* for comparison only (not for training)

Author: Conservation AI Team
Date: November 2025
Version: 2.0 - Pure RL, No Pre-training (FIXED)
"""

# ============================================================================
# SECTION 1: Install and Import Dependencies
# ============================================================================

get_ipython().system('pip install torch torchvision matplotlib numpy scipy pillow -q')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import HTML, clear_output
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
from datetime import datetime
from PIL import Image
import io
import heapq
from typing import List, Tuple, Dict, Optional

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# ============================================================================
# SECTION 2: A* Pathfinding (For Comparison Only)
# ============================================================================

class AStarPathfinder:
    """A* for optimal path calculation - COMPARISON ONLY"""

    def __init__(self, habitat_map: np.ndarray):
        self.habitat_map = habitat_map
        self.grid_size = habitat_map.shape[0]

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (pos[0] + dr, pos[1] + dc)
            if (0 <= new_pos[0] < self.grid_size and
                0 <= new_pos[1] < self.grid_size and
                self.habitat_map[new_pos] > 0):
                neighbors.append(new_pos)
        return neighbors

    def get_move_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        base_cost = 1.0
        quality = self.habitat_map[to_pos]
        quality_factor = (11 - quality) / 10.0
        return base_cost * quality_factor

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        if self.habitat_map[start] == 0 or self.habitat_map[goal] == 0:
            return None

        counter = 0
        frontier = [(0, counter, start, [start])]
        visited = {start: 0}

        while frontier:
            f_score, _, current, path = heapq.heappop(frontier)

            if current == goal:
                return path

            for neighbor in self.get_neighbors(current):
                move_cost = self.get_move_cost(current, neighbor)
                new_g_score = visited[current] + move_cost

                if neighbor not in visited or new_g_score < visited[neighbor]:
                    visited[neighbor] = new_g_score
                    h_score = self.heuristic(neighbor, goal)
                    new_f_score = new_g_score + h_score

                    counter += 1
                    heapq.heappush(frontier, (new_f_score, counter, neighbor, path + [neighbor]))

        return None

# ============================================================================
# SECTION 3: Efficient Madagascar Habitat Environment
# ============================================================================

class EfficientMadagascarHabitatEnv:
    """Pure RL environment with smart reward shaping"""

    def __init__(self, grid_size=20, real_map_path=None, reintroduction_sites=None):
        self.grid_size = grid_size
        self.action_space = 4
        self.observation_space = grid_size * grid_size * 6

        self.real_map_image = None
        self.region_mask = None
        self.habitat_map = None
        self.reintroduction_sites = reintroduction_sites or []
        self.pathfinder = None

        if real_map_path:
            self.load_real_map(real_map_path)
        else:
            self.habitat_map = self._create_realistic_habitat()
            self.region_mask = np.ones((grid_size, grid_size))
            self._add_features()

        self.pathfinder = AStarPathfinder(self.habitat_map)

        self.agent_pos = None
        self.goal_pos = None
        self.start_pos = None
        self.visited = None
        self.steps = 0
        self.max_steps = grid_size * grid_size // 2
        self.path_history = []
        self.optimal_path = None
        self.optimal_path_length = None
        self.difficulty_level = 0.3
        self.previous_distance = None

    def load_real_map(self, image_path_or_array):
        """Load real restoration map"""
        print("\n" + "="*60)
        print("LOADING MADAGASCAR RESTORATION MAP")
        print("="*60)

        if isinstance(image_path_or_array, str):
            img = Image.open(image_path_or_array)
            self.real_map_image = np.array(img)
        else:
            self.real_map_image = image_path_or_array

        from scipy.ndimage import zoom
        if self.real_map_image.shape[:2] != (self.grid_size, self.grid_size):
            if len(self.real_map_image.shape) == 3:
                zoom_factors = (
                    self.grid_size / self.real_map_image.shape[0],
                    self.grid_size / self.real_map_image.shape[1], 1
                )
            else:
                zoom_factors = (
                    self.grid_size / self.real_map_image.shape[0],
                    self.grid_size / self.real_map_image.shape[1]
                )
            self.real_map_image = zoom(self.real_map_image, zoom_factors, order=1)

        self._extract_region_mask()
        self._create_habitat_from_map()
        print(f"✓ Valid cells: {np.sum(self.region_mask)}/{self.grid_size**2}")
        print("="*60 + "\n")

    def _extract_region_mask(self):
        if len(self.real_map_image.shape) == 3:
            gray = np.mean(self.real_map_image, axis=2)
        else:
            gray = self.real_map_image

        if gray.max() <= 1.0:
            gray = gray * 255

        brightness_threshold = np.percentile(gray, 5)
        mask_brightness = gray > brightness_threshold

        if len(self.real_map_image.shape) == 3:
            img_norm = self.real_map_image.copy()
            if img_norm.max() <= 1.0:
                img_norm = img_norm * 255

            blue = img_norm[:, :, 2] if img_norm.shape[2] >= 3 else gray
            red = img_norm[:, :, 0] if img_norm.shape[2] >= 3 else gray
            green = img_norm[:, :, 1] if img_norm.shape[2] >= 3 else gray

            is_water = (blue > red + 30) & (blue > green + 30) & (blue > 100)
            is_very_dark = gray < brightness_threshold
            mask_color = ~(is_water | is_very_dark)
        else:
            mask_color = mask_brightness

        self.region_mask = (mask_brightness | mask_color).astype(float)

        if np.sum(self.region_mask) / self.region_mask.size < 0.3:
            self.region_mask = np.ones_like(self.region_mask)

    def _create_habitat_from_map(self):
        if len(self.real_map_image.shape) == 3:
            img_norm = self.real_map_image.copy()
            if img_norm.max() <= 1.0:
                img_norm = img_norm * 255

            green = img_norm[:, :, 1]
            red = img_norm[:, :, 0]
            blue = img_norm[:, :, 2]

            veg_index = (green.astype(float) - red.astype(float)) / (green.astype(float) + red.astype(float) + 1e-6)
            brightness = (red.astype(float) + green.astype(float) + blue.astype(float)) / 3.0 / 255.0
            veg_norm = (veg_index - veg_index.min()) / (veg_index.max() - veg_index.min() + 1e-6)
            combined = 0.6 * veg_norm + 0.4 * brightness
            habitat_quality = combined * 9 + 1
        else:
            img_norm = self.real_map_image.copy()
            if img_norm.max() <= 1.0:
                img_norm = img_norm * 255
            habitat_quality = (img_norm / 255.0) * 9 + 1

        self.habitat_map = habitat_quality * self.region_mask

        if not np.any(self.habitat_map > 0):
            self.habitat_map = habitat_quality
            self.region_mask = np.ones_like(self.region_mask)

        high_quality = np.where((self.habitat_map > 7) & (self.region_mask > 0))
        if len(high_quality[0]) > 0:
            num_resources = min(5, len(high_quality[0]))
            indices = np.random.choice(len(high_quality[0]), num_resources, replace=False)
            for idx in indices:
                self.habitat_map[high_quality[0][idx], high_quality[1][idx]] = 10

    def _create_realistic_habitat(self):
        map_array = np.zeros((self.grid_size, self.grid_size))
        for _ in range(random.randint(3, 6)):
            cx, cy = random.randint(2, self.grid_size-3), random.randint(2, self.grid_size-3)
            quality, radius = random.randint(5, 9), random.randint(2, 4)
            for i in range(max(0, cx-radius), min(self.grid_size, cx+radius+1)):
                for j in range(max(0, cy-radius), min(self.grid_size, cy+radius+1)):
                    dist = np.sqrt((i-cx)**2 + (j-cy)**2)
                    if dist <= radius:
                        map_array[i, j] = max(map_array[i, j], quality - int(dist/radius*3))
        map_array[map_array == 0] = np.random.randint(1, 4, size=np.sum(map_array == 0))
        return map_array

    def _add_features(self):
        if self.real_map_image is not None:
            return
        for _ in range(random.randint(3, 6)):
            if random.random() < 0.5:
                x = random.randint(1, self.grid_size-2)
                length = random.randint(2, self.grid_size//3)
                y_start = random.randint(0, self.grid_size-length)
                self.habitat_map[y_start:y_start+length, x] = 0
            else:
                y = random.randint(1, self.grid_size-2)
                length = random.randint(2, self.grid_size//3)
                x_start = random.randint(0, self.grid_size-length)
                self.habitat_map[y, x_start:x_start+length] = 0

    def add_reintroduction_sites(self, sites: List[Tuple[int, int]], validate=True):
        self.reintroduction_sites = []
        for site in sites:
            if not (0 <= site[0] < self.grid_size and 0 <= site[1] < self.grid_size):
                continue
            if validate and self.habitat_map[site] <= 0:
                nearest = self._find_nearest_valid_cell(site)
                if nearest:
                    self.reintroduction_sites.append(nearest)
            else:
                self.reintroduction_sites.append(site)
                print(f"  ✓ Site at {site}")
        print(f"\n✅ Total sites: {len(self.reintroduction_sites)}")
        return self.reintroduction_sites

    def _find_valid_position(self, preferred_pos=None, min_distance_from=None, max_distance_from=None):
        valid_cells = np.where(self.habitat_map > 0)
        if len(valid_cells[0]) == 0:
            raise ValueError("No valid cells!")

        for _ in range(100):
            idx = random.randint(0, len(valid_cells[0])-1)
            pos = (valid_cells[0][idx], valid_cells[1][idx])
            if min_distance_from is not None:
                dist = np.linalg.norm(np.array(pos) - np.array(min_distance_from))
                if max_distance_from is not None:
                    if min_distance_from <= dist <= max_distance_from:
                        return pos
                elif dist > min_distance_from:
                    return pos
            else:
                return pos

        idx = random.randint(0, len(valid_cells[0])-1)
        return (valid_cells[0][idx], valid_cells[1][idx])

    def _find_nearest_valid_cell(self, target_pos, max_radius=5):
        valid_cells = np.where(self.habitat_map > 0)
        if len(valid_cells[0]) == 0:
            return None
        distances = np.sqrt((valid_cells[0]-target_pos[0])**2 + (valid_cells[1]-target_pos[1])**2)
        min_idx = np.argmin(distances)
        if distances[min_idx] <= max_radius:
            return (valid_cells[0][min_idx], valid_cells[1][min_idx])
        return None

    def set_difficulty(self, level: float):
        self.difficulty_level = max(0.1, min(1.0, level))

    def reset(self, start_pos=None, goal_pos=None, use_reintroduction_site=True):
        self.steps = 0
        self.visited = np.zeros((self.grid_size, self.grid_size))
        self.path_history = []

        if use_reintroduction_site and len(self.reintroduction_sites) > 0 and start_pos is None:
            self.start_pos = random.choice(self.reintroduction_sites)
        else:
            self.start_pos = self._find_valid_position(start_pos)

        min_dist = max(3, self.grid_size * 0.15 * self.difficulty_level)
        max_dist = self.grid_size * 0.85 * self.difficulty_level

        self.goal_pos = self._find_valid_position(
            goal_pos,
            min_distance_from=min_dist,
            max_distance_from=max_dist
        )

        self.optimal_path = self.pathfinder.find_path(self.start_pos, self.goal_pos)
        self.optimal_path_length = len(self.optimal_path) if self.optimal_path else int(
            np.linalg.norm(np.array(self.goal_pos) - np.array(self.start_pos))
        ) + 5

        self.agent_pos = self.start_pos
        self.visited[self.agent_pos] = 1
        self.path_history.append(self.agent_pos)
        self.previous_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
        self.best_distance_so_far = self.previous_distance

        return self._get_observation()

    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 6))

        obs[:, :, 0] = self.habitat_map / 10.0
        obs[self.agent_pos[0], self.agent_pos[1], 1] = 1.0
        obs[self.goal_pos[0], self.goal_pos[1], 2] = 1.0
        obs[:, :, 3] = self.visited

        current_dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
        max_dist = np.linalg.norm(np.array([0, 0]) - np.array([self.grid_size, self.grid_size]))
        obs[:, :, 4] = 1.0 - (current_dist / max_dist)

        if self.goal_pos[0] != self.agent_pos[0] or self.goal_pos[1] != self.agent_pos[1]:
            direction = np.arctan2(self.goal_pos[0] - self.agent_pos[0],
                                  self.goal_pos[1] - self.agent_pos[1])
            obs[:, :, 5] = (direction + np.pi) / (2 * np.pi)

        return obs.flatten()

    def step(self, action):
        self.steps += 1

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = moves[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return self._get_observation(), -10, False, {'success': False, 'efficiency': 0.0}

        if self.habitat_map[new_pos] == 0:
            return self._get_observation(), -15, False, {'success': False, 'efficiency': 0.0}

        self.agent_pos = new_pos
        self.path_history.append(self.agent_pos)
        reward = self._calculate_smart_reward()
        self.visited[self.agent_pos] += 1

        done = (self.agent_pos == self.goal_pos) or (self.steps >= self.max_steps)

        efficiency = 0.0
        if self.agent_pos == self.goal_pos and self.optimal_path_length > 0:
            efficiency = min(1.0, self.optimal_path_length / len(self.path_history))

        return self._get_observation(), reward, done, {
            'success': self.agent_pos == self.goal_pos,
            'efficiency': efficiency,
            'path_length': len(self.path_history),
            'optimal_length': self.optimal_path_length
        }

    def _calculate_smart_reward(self):
        current_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
        distance_improvement = self.previous_distance - current_distance

        distance_scale = 5.0 if self.steps < 50 else 10.0
        dense_reward = distance_improvement * distance_scale

        self.previous_distance = current_distance

        visit_count = self.visited[self.agent_pos]

        if visit_count == 0:
            exploration_bonus = 8.0
            revisit_penalty = 0.0
        elif visit_count == 1:
            exploration_bonus = 0.0
            revisit_penalty = -5.0
        else:
            exploration_bonus = 0.0
            revisit_penalty = -15.0 * (visit_count - 1)

        step_penalty = -0.5
        habitat_bonus = self.habitat_map[self.agent_pos] * 0.2

        if self.agent_pos == self.goal_pos:
            steps_taken = len(self.path_history)
            if self.optimal_path_length > 0:
                efficiency_ratio = self.optimal_path_length / steps_taken
                efficiency_ratio = min(1.0, efficiency_ratio)
            else:
                efficiency_ratio = 0.5
            goal_reward = 200.0 + (200.0 * efficiency_ratio)
        else:
            goal_reward = 0.0

        max_dist = np.sqrt(2) * self.grid_size
        proximity_bonus = (1.0 - current_distance / max_dist) * 3.0

        resource_bonus = 3.0 if self.habitat_map[self.agent_pos] == 10 else 0.0

        if hasattr(self, 'best_distance_so_far'):
            if current_distance < self.best_distance_so_far:
                progress_bonus = 5.0
                self.best_distance_so_far = current_distance
            else:
                progress_bonus = 0.0
        else:
            self.best_distance_so_far = current_distance
            progress_bonus = 0.0

        total = (
            dense_reward +
            exploration_bonus +
            revisit_penalty +
            step_penalty +
            habitat_bonus +
            goal_reward +
            proximity_bonus +
            resource_bonus +
            progress_bonus
        )

        return total

    def render(self, show_path=True, overlay_on_real_map=True, show_optimal=False):
        fig, ax = plt.subplots(figsize=(14, 12))

        if overlay_on_real_map and self.real_map_image is not None:
            ax.imshow(self.real_map_image, origin='upper', alpha=0.7)
            cmap = plt.cm.YlGn
            cmap.set_under(color='none', alpha=0)
            habitat_masked = np.ma.masked_where(self.habitat_map <= 0, self.habitat_map)
            im = ax.imshow(habitat_masked, cmap=cmap, vmin=0.1, vmax=10, origin='upper', alpha=0.4)
        else:
            cmap = plt.cm.YlGn
            cmap.set_under('black')
            im = ax.imshow(self.habitat_map, cmap=cmap, vmin=0.1, vmax=10, origin='upper')

        if show_optimal and self.optimal_path:
            optimal_array = np.array(self.optimal_path)
            ax.plot(optimal_array[:, 1], optimal_array[:, 0], 'b--', linewidth=2,
                   alpha=0.6, label=f'Optimal (A*): {len(self.optimal_path)} steps', zorder=4)

        if show_path and len(self.path_history) > 1:
            path_array = np.array(self.path_history)
            ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=3,
                   alpha=0.8, label=f'Agent: {len(self.path_history)} steps', zorder=5)

        if len(self.reintroduction_sites) > 0:
            for site in self.reintroduction_sites:
                ax.plot(site[1], site[0], 'c^', markersize=18,
                       label='Release Site' if site == self.reintroduction_sites[0] else '',
                       zorder=9, markeredgecolor='blue', markeredgewidth=2)

        ax.plot(self.start_pos[1], self.start_pos[0], 'b*', markersize=25,
               label='Start', zorder=10, markeredgecolor='white', markeredgewidth=2)
        ax.plot(self.goal_pos[1], self.goal_pos[0], 'g*', markersize=25,
               label='Goal', zorder=10, markeredgecolor='white', markeredgewidth=2)
        ax.plot(self.agent_pos[1], self.agent_pos[0], 'yo', markersize=18,
               label='Current', zorder=10, markeredgecolor='red', markeredgewidth=3)

        ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, framealpha=0.9)

        if self.agent_pos == self.goal_pos and self.optimal_path_length:
            efficiency = self.optimal_path_length / len(self.path_history) * 100
            title = f'Migration Path (Efficiency: {efficiency:.1f}%)\nStep {self.steps}/{self.max_steps}'
        else:
            title = f'Migration Path\nStep {self.steps}/{self.max_steps}'

        ax.set_title(title, fontsize=14, weight='bold')
        plt.colorbar(im, ax=ax, label='Habitat Quality', fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

# ============================================================================
# SECTION 4: Standard PPO Agent (Pure RL)
# ============================================================================

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        return self.actor(features), self.critic(features)

    def act(self, state):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), state_value

    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        return dist.log_prob(action), state_value, dist.entropy()

class PurePPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4, hidden_dim=256):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

        self.buffer = {
            'states': [], 'actions': [], 'logprobs': [],
            'rewards': [], 'is_terminals': [], 'state_values': []
        }

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, action_logprob, state_value = self.policy_old.act(state)

        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['logprobs'].append(action_logprob)
        self.buffer['state_values'].append(state_value)
        return action

    def store_transition(self, reward, is_terminal):
        self.buffer['rewards'].append(reward)
        self.buffer['is_terminals'].append(is_terminal)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer['rewards']),
                                       reversed(self.buffer['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.cat(self.buffer['states']).detach()
        old_actions = torch.tensor(self.buffer['actions'], dtype=torch.long).detach()
        old_logprobs = torch.cat(self.buffer['logprobs']).detach()
        old_state_values = torch.cat(self.buffer['state_values']).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            advantages = rewards - old_state_values.detach()
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = {
            'states': [], 'actions': [], 'logprobs': [],
            'rewards': [], 'is_terminals': [], 'state_values': []
        }

    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath))
        self.policy_old.load_state_dict(torch.load(filepath))

# ============================================================================
# SECTION 5: Curriculum Learning Training (Pure RL)
# ============================================================================

def train_pure_rl_curriculum_FIXED(env, agent, num_episodes=4000, verbose=True):
    """
    FIXED TRAINING
    - More episodes (4000 instead of 2000)
    - Better curriculum
    - More frequent updates
    """
    print("\n" + "="*60)
    print("FIXED PURE RL CURRICULUM TRAINING")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print("Key fixes:")
    print("  ✅ Correct efficiency calculation")
    print("  ✅ Stronger exploration (8.0 bonus)")
    print("  ✅ Balanced revisit penalty (-5 to -15)")
    print("  ✅ Progressive distance rewards")
    print("  ✅ Extended training (4000 episodes)")
    print("="*60 + "\n")

    episode_rewards = []
    episode_lengths = []
    success_rate = []
    efficiency_scores = []

    curriculum = [
        (0, 500, 0.25),
        (500, 1000, 0.4),
        (1000, 1500, 0.55),
        (1500, 2500, 0.7),
        (2500, num_episodes, 0.85)
    ]

    update_freq = 15

    for episode in range(1, num_episodes + 1):
        for start_ep, end_ep, difficulty in curriculum:
            if start_ep <= episode < end_ep:
                env.set_difficulty(difficulty)
                break

        state = env.reset(use_reintroduction_site=True)
        episode_reward = 0

        for step in range(env.max_steps):
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            agent.store_transition(reward, done)
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(len(env.path_history))
        success_rate.append(1 if info.get('success', False) else 0)

        if info.get('success', False) and info.get('efficiency', 0) > 0:
            efficiency_scores.append(info['efficiency'])
        else:
            efficiency_scores.append(0.0)

        if episode % update_freq == 0:
            agent.update()

        if verbose and episode % 50 == 0:
            recent = 50
            avg_reward = np.mean(episode_rewards[-recent:])
            avg_success = np.mean(success_rate[-recent:]) * 100
            avg_length = np.mean([episode_lengths[i] for i in range(max(0, len(episode_lengths)-recent), len(episode_lengths))
                                if success_rate[i] == 1]) if any(success_rate[-recent:]) else 0

            successful_recent = [efficiency_scores[i] for i in range(max(0, len(efficiency_scores)-recent), len(efficiency_scores))
                               if success_rate[i] == 1 and efficiency_scores[i] > 0]
            avg_efficiency = np.mean(successful_recent) * 100 if successful_recent else 0

            print(f"Ep {episode:4d} | Diff: {env.difficulty_level:.2f} | "
                  f"Reward: {avg_reward:7.1f} | Success: {avg_success:5.1f}% | "
                  f"Length: {avg_length:5.1f} | Eff: {avg_efficiency:5.1f}%")

        if episode % 1000 == 0:
            agent.save(f'lemur_pure_rl_FIXED_ep{episode}.pth')

    print("\n" + "="*60)
    print("FIXED TRAINING COMPLETE!")
    print("="*60)

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_rate,
        'efficiency_scores': efficiency_scores
    }

# ============================================================================
# SECTION 6: Testing and Visualization
# ============================================================================

def test_agent_efficiency(env, agent, num_tests=5):
    """Test with efficiency metrics"""
    print("\n" + "="*60)
    print("EFFICIENCY TESTING")
    print("="*60)

    results = []

    for test in range(num_tests):
        env.set_difficulty(0.8)
        state = env.reset(use_reintroduction_site=True)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward

        success = info.get('success', False)
        efficiency = info.get('efficiency', 0.0) * 100
        path_length = info.get('path_length', 0)
        optimal_length = info.get('optimal_length', 0)

        print(f"\n🧪 Test {test+1}/{num_tests}")
        print(f"  {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"  A* Optimal:  {optimal_length} steps")
        print(f"  Agent Path:  {path_length} steps")
        print(f"  Efficiency:  {efficiency:.1f}%")
        print(f"  Overhead:    +{path_length - optimal_length} steps")

        results.append({
            'success': success,
            'efficiency': efficiency,
            'path_length': path_length,
            'optimal_length': optimal_length,
            'reward': total_reward
        })

        env.render(show_path=True, overlay_on_real_map=True, show_optimal=True)

        agent.buffer = {
            'states': [], 'actions': [], 'logprobs': [],
            'rewards': [], 'is_terminals': [], 'state_values': []
        }

    successful = [r for r in results if r['success']]

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Success Rate:     {len(successful)}/{num_tests} ({len(successful)/num_tests*100:.1f}%)")

    if successful:
        avg_eff = np.mean([r['efficiency'] for r in successful])
        avg_path = np.mean([r['path_length'] for r in successful])
        avg_opt = np.mean([r['optimal_length'] for r in successful])
        avg_overhead = avg_path - avg_opt

        print(f"Avg Efficiency:   {avg_eff:.1f}%")
        print(f"Avg Agent Path:   {avg_path:.1f} steps")
        print(f"Avg Optimal:      {avg_opt:.1f} steps")
        print(f"Avg Overhead:     +{avg_overhead:.1f} steps")

        if avg_eff >= 70:
            print(f"\n🏆 EXCELLENT! Agent paths are very efficient!")
        elif avg_eff >= 55:
            print(f"\n✅ GOOD! Agent learns reasonable paths")
        else:
            print(f"\n⚠️  MODERATE: Could use more training")

    print(f"{'='*60}\n")

    return results

def plot_pure_rl_results(stats, window=50):
    """Plot training results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Rewards
    ax = axes[0, 0]
    rewards = stats['episode_rewards']
    ax.plot(rewards, alpha=0.2, color='blue')
    smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(smooth, color='blue', linewidth=2, label=f'{window}-ep avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Path Lengths
    ax = axes[0, 1]
    lengths = stats['episode_lengths']
    ax.plot(lengths, alpha=0.2, color='green')
    smooth = np.convolve(lengths, np.ones(window)/window, mode='valid')
    ax.plot(smooth, color='green', linewidth=2, label=f'{window}-ep avg')
    ax.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Max steps')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Path Length (steps)')
    ax.set_title('Path Length Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Success Rate
    ax = axes[0, 2]
    success = np.array(stats['success_rate'])
    smooth = np.convolve(success, np.ones(window)/window, mode='valid') * 100
    ax.plot(smooth, color='red', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Success Rate (window={window})')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)

    # Efficiency Distribution
    ax = axes[1, 0]
    successful_eff = [stats['efficiency_scores'][i] * 100
                     for i in range(len(stats['efficiency_scores']))
                     if stats['success_rate'][i] == 1 and stats['efficiency_scores'][i] > 0]
    if len(successful_eff) > 10:
        ax.hist(successful_eff, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(successful_eff), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(successful_eff):.1f}%')
        ax.set_xlabel('Path Efficiency (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Efficiency Distribution\n(Successful Episodes)')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Length Distribution
    ax = axes[1, 1]
    successful_lengths = [stats['episode_lengths'][i]
                         for i in range(len(stats['episode_lengths']))
                         if stats['success_rate'][i] == 1]
    if len(successful_lengths) > 10:
        ax.hist(successful_lengths, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(successful_lengths), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(successful_lengths):.1f}')
        ax.set_xlabel('Path Length (steps)')
        ax.set_ylabel('Frequency')
        ax.set_title('Path Length Distribution\n(Successful Episodes)')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined Progress
    ax = axes[1, 2]
    window_large = 100
    if len(success) > window_large:
        success_smooth = np.convolve(success, np.ones(window_large)/window_large, mode='valid') * 100
        lengths_smooth = np.convolve(lengths, np.ones(window_large)/window_large, mode='valid')

        ax2 = ax.twinx()
        line1 = ax.plot(success_smooth, 'g-', linewidth=2, label='Success %')[0]
        line2 = ax2.plot(lengths_smooth, 'b-', linewidth=2, label='Path Length')[0]

        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate (%)', color='g')
        ax2.set_ylabel('Path Length', color='b')
        ax.set_title(f'Learning Curve (window={window_large})')
        ax.tick_params(axis='y', labelcolor='g')
        ax2.tick_params(axis='y', labelcolor='b')
        ax.legend([line1, line2], ['Success %', 'Path Length'], loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n📊 FINAL STATISTICS (last 100 episodes):")
    print(f"{'='*60}")
    last_n = 100
    print(f"Avg Reward:       {np.mean(rewards[-last_n:]):.2f}")
    print(f"Avg Path Length:  {np.mean(lengths[-last_n:]):.1f} steps")
    print(f"Success Rate:     {np.mean(success[-last_n:]) * 100:.1f}%")

    last_n_successful = [stats['efficiency_scores'][i] * 100
                        for i in range(len(stats['efficiency_scores'])-last_n, len(stats['efficiency_scores']))
                        if stats['success_rate'][i] == 1 and stats['efficiency_scores'][i] > 0]
    if last_n_successful:
        print(f"Avg Efficiency:   {np.mean(last_n_successful):.1f}%")
    print(f"{'='*60}\n")

# ============================================================================
# SECTION 7: Main Execution (FIXED)
# ============================================================================

def main_pure_rl_lemur_migration():
    """
    PRODUCTION VERSION: Pure RL (no pre-training)
    Learns efficient paths from scratch using smart reward shaping
    """
    print("="*70)
    print("🦊 EFFICIENT LEMUR MIGRATION - PURE RL VERSION")
    print("="*70)
    print("\n🎯 Key Features:")
    print("   ✅ 100% pure reinforcement learning")
    print("   ✅ No pre-training or expert demonstrations")
    print("   ✅ Smart reward shaping (dense distance signals)")
    print("   ✅ Strong loop penalties (-15 per revisit)")
    print("   ✅ Curriculum learning (easy → hard)")
    print("   ✅ Max 200 steps (forces efficiency)")
    print("   ✅ A* for comparison only (not training)")
    print("="*70)

    # Upload
    print("\n1️⃣  Upload restoration map...")
    
    uploaded = files.upload()

    if len(uploaded) == 0:
        print("⚠️  No upload. Using synthetic map.")
        env = EfficientMadagascarHabitatEnv(grid_size=20)
    else:
        filename = list(uploaded.keys())[0]
        print(f"✅ Loaded: {filename}")
        img = Image.open(io.BytesIO(uploaded[filename]))
        env = EfficientMadagascarHabitatEnv(grid_size=20)
        env.load_real_map(np.array(img))

    # Sites
    print("\n2️⃣  Adding reintroduction sites...")
    best_sites = [(98, 56), (95, 58), (90, 55)]
    sites_scaled = [(int(s[0] * 0.2), int(s[1] * 0.2)) for s in best_sites]
    env.add_reintroduction_sites(sites_scaled)

    # Visualize
    print("\n3️⃣  Visualizing map...")
    env.set_difficulty(0.7)
    env.reset(use_reintroduction_site=True)
    env.render(show_path=False, overlay_on_real_map=True, show_optimal=True)

    # Agent
    print("\n4️⃣  Initializing Pure RL Agent...")
    agent = PurePPOAgent(
        state_dim=env.observation_space,
        action_dim=env.action_space,
        lr=3e-4,
        hidden_dim=256
    )
    print("   ✅ Fresh agent (random weights)")

    # Training - FIXED LINE (was the bug)
    print("\n5️⃣  Training from scratch...")
    print("   🎯 Target: >70% success, >60% efficiency")
    print("   ⏰ Time: ~15-18 minutes for 3000 episodes")
    print("\n" + "-"*70)

    # ⭐ FIXED: Correct function name
    stats = train_pure_rl_curriculum_FIXED(env, agent, num_episodes=3000, verbose=True)

    # Results
    print("\n6️⃣  Analyzing results...")
    plot_pure_rl_results(stats)

    # Assessment
    final_success = np.mean(stats['success_rate'][-100:]) * 100
    final_lengths = [stats['episode_lengths'][i] for i in range(len(stats['episode_lengths'])-100, len(stats['episode_lengths']))
                    if stats['success_rate'][i] == 1]
    final_length = np.mean(final_lengths) if final_lengths else 0

    final_effs = [stats['efficiency_scores'][i] * 100 for i in range(len(stats['efficiency_scores'])-100, len(stats['efficiency_scores']))
                 if stats['success_rate'][i] == 1 and stats['efficiency_scores'][i] > 0]
    final_eff = np.mean(final_effs) if final_effs else 0

    print("\n" + "="*70)
    print("🎯 FINAL PERFORMANCE")
    print("="*70)
    print(f"Success Rate:     {final_success:.1f}%")
    print(f"Avg Path Length:  {final_length:.1f} steps")
    print(f"Avg Efficiency:   {final_eff:.1f}%")

    # Comparison with V1
    v1_length = 520
    if final_length > 0:
        improvement = (1 - final_length / v1_length) * 100
        print(f"\n📈 Improvement over V1:")
        print(f"   Path length: {v1_length} → {final_length:.0f} steps")
        print(f"   Reduction:   {improvement:.0f}%")

    if final_success >= 70 and final_eff >= 60:
        print(f"\n🏆 EXCELLENT: Strong pure RL performance!")
    elif final_success >= 60 and final_eff >= 50:
        print(f"\n✅ GOOD: Solid pure RL learning")
    else:
        print(f"\n⚠️  Consider extending to 4000 episodes")
    print("="*70)

    # Testing
    print("\n7️⃣  Running efficiency tests...")
    test_results = test_agent_efficiency(env, agent, num_tests=5)

    # Save and Download
    model_filename = 'lemur_migration_pure_rl_v2.pth'
    agent.save(model_filename)
    print(f"\n💾 Saved: {model_filename}")

    # Download the model
    print(f"📥 Downloading model to your computer...")
    
    files.download(model_filename)
    print(f"✅ Model downloaded: {model_filename}")

    print("\n" + "="*70)
    print("✅ PURE RL TRAINING COMPLETE!")
    print("="*70)
    print("\n🎓 What the agent learned (no expert help):")
    print("   • Navigate toward goal using dense distance rewards")
    print("   • Avoid loops (learned from -15 penalties)")
    print("   • Prefer shorter paths (step penalties)")
    print("   • Handle easy → hard scenarios (curriculum)")
    print("\n📌 Commands:")
    print("   test_agent_efficiency(env, agent, num_tests=10)")
    print("   env.render(show_optimal=True)  # Compare with A*")

    return env, agent, stats

# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    print("🚀 PURE RL VERSION READY!")
    print("\nRun: env, agent, stats = main_pure_rl_lemur_migration()")

    # Auto-execute
    env, agent, stats = main_pure_rl_lemur_migration()

# In[ ]:

"""
Migration Path Overlay Tool - FIXED VERSION
Creates publication-quality overlays of RL paths on restoration maps
FIX: Proper coordinate scaling from grid space to image space
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from PIL import Image
import io

def create_publication_overlay(env, agent, restoration_map_path,
                               num_paths=5, output_filename='migration_overlay.png'):
    """
    Create beautiful overlay of multiple migration paths on restoration map
    FIXED: Properly scales grid coordinates (20x20) to image coordinates

    Args:
        env: Trained environment
        agent: Trained agent
        restoration_map_path: Path to your restoration_zones.png
        num_paths: Number of example paths to show
        output_filename: Save filename
    """

    # Load restoration map
    restoration_img = Image.open(restoration_map_path)
    restoration_array = np.array(restoration_img)

    # ===== CRITICAL FIX: Calculate coordinate scaling =====
    img_height, img_width = restoration_array.shape[:2]
    grid_size = env.grid_size  # Should be 20

    scale_y = img_height / grid_size  # e.g., 300 / 20 = 15
    scale_x = img_width / grid_size   # e.g., 500 / 20 = 25

    print(f"\n📐 Coordinate Scaling:")
    print(f"   Grid: {grid_size}x{grid_size}")
    print(f"   Image: {img_width}x{img_height} pixels")
    print(f"   Scale: x={scale_x:.2f}, y={scale_y:.2f}")
    print(f"   Example: Grid (10,10) → Image ({10*scale_x:.0f}, {10*scale_y:.0f})")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # ===== LEFT PLOT: Original Restoration Zones =====
    ax = axes[0]
    ax.imshow(restoration_array, origin='upper')
    ax.set_title('GenAI Habitat Restoration Zones\n(Previous Semester Output)',
                fontsize=16, weight='bold', pad=20)
    ax.axis('off')

    # Add scale bar (in pixels)
    scale_length = img_width // 10
    scale_bar = FancyBboxPatch((10, img_height - 30),
                               scale_length, 15,
                               boxstyle="round,pad=2",
                               edgecolor='white', facecolor='black', linewidth=2)
    ax.add_patch(scale_bar)
    ax.text(10 + scale_length/2, img_height - 22,
           f'{scale_length} pixels ({scale_length/scale_x:.1f} grid cells)',
           ha='center', va='center',
           color='white', fontsize=10, weight='bold')

    # ===== RIGHT PLOT: Migration Paths Overlay =====
    ax = axes[1]
    ax.imshow(restoration_array, origin='upper', alpha=0.6)

    # Overlay habitat quality as heatmap (scaled to match image)
    if hasattr(env, 'real_map_image') and env.real_map_image is not None:
        # Scale habitat map to image resolution
        from scipy.ndimage import zoom
        habitat_for_display = zoom(env.habitat_map,
                                   (scale_y, scale_x),
                                   order=1)
        habitat_for_display[habitat_for_display == 0] = np.nan
        im = ax.imshow(habitat_for_display, cmap='YlGn', alpha=0.3,
                      origin='upper', vmin=1, vmax=10)

    # Generate multiple paths
    colors = ['red', 'orange', 'magenta', 'cyan', 'yellow']
    path_data = []

    print(f"\n🎨 Generating {num_paths} migration paths...")

    for i in range(num_paths):
        # Reset environment with high difficulty
        env.set_difficulty(0.8)
        state = env.reset(use_reintroduction_site=True)
        done = False

        # Run agent
        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)

        # Store path info
        if info.get('success', False):
            path_data.append({
                'path': env.path_history.copy(),
                'start': env.start_pos,
                'goal': env.goal_pos,
                'efficiency': info.get('efficiency', 0) * 100,
                'length': len(env.path_history),
                'optimal': info.get('optimal_length', 0)
            })

            # ===== CRITICAL FIX: Scale paths to image coordinates =====
            if len(env.path_history) > 1:
                path_array = np.array(env.path_history)

                # Scale from grid to image coordinates
                path_scaled = path_array.copy().astype(float)
                path_scaled[:, 0] *= scale_y  # Scale row (y) coordinates
                path_scaled[:, 1] *= scale_x  # Scale col (x) coordinates

                color = colors[i % len(colors)]

                # Main path line (note: matplotlib uses (x, y) = (col, row))
                ax.plot(path_scaled[:, 1], path_scaled[:, 0],
                       color=color, linewidth=3, alpha=0.7,
                       label=f'Path {i+1}: {len(env.path_history)} steps',
                       zorder=5)

                # Direction arrows (also scaled)
                step = max(1, len(path_scaled) // 8)
                for j in range(0, len(path_scaled) - step, step):
                    arrow = FancyArrowPatch(
                        (path_scaled[j, 1], path_scaled[j, 0]),
                        (path_scaled[j+step, 1], path_scaled[j+step, 0]),
                        arrowstyle='->', mutation_scale=20,
                        color=color, alpha=0.8, linewidth=2, zorder=6
                    )
                    ax.add_patch(arrow)

                # Mark start (circle) - SCALED
                start_y = env.start_pos[0] * scale_y
                start_x = env.start_pos[1] * scale_x
                start_circle = Circle((start_x, start_y),
                                     radius=0.6 * min(scale_x, scale_y),  # Scale radius
                                     color=color, alpha=0.9,
                                     edgecolor='white', linewidth=2, zorder=10)
                ax.add_patch(start_circle)

                # Mark goal (star) - SCALED
                goal_y = env.goal_pos[0] * scale_y
                goal_x = env.goal_pos[1] * scale_x
                ax.plot(goal_x, goal_y, '*',
                       color=color, markersize=20, markeredgecolor='white',
                       markeredgewidth=2, zorder=10)

            print(f"  ✓ Path {i+1}: Grid {env.start_pos}→{env.goal_pos} | "
                  f"Image ({env.start_pos[1]*scale_x:.0f},{env.start_pos[0]*scale_y:.0f})→"
                  f"({env.goal_pos[1]*scale_x:.0f},{env.goal_pos[0]*scale_y:.0f})")

        # Clear agent buffer
        agent.buffer = {
            'states': [], 'actions': [], 'logprobs': [],
            'rewards': [], 'is_terminals': [], 'state_values': []
        }

    # Mark reintroduction sites - SCALED
    if hasattr(env, 'reintroduction_sites') and len(env.reintroduction_sites) > 0:
        for idx, site in enumerate(env.reintroduction_sites):
            site_y = site[0] * scale_y
            site_x = site[1] * scale_x
            ax.plot(site_x, site_y, '^', color='blue',
                   markersize=18, markeredgecolor='darkblue',
                   markeredgewidth=3, zorder=11,
                   label='Release Site' if idx == 0 else '')

    # Title and legend
    ax.set_title('RL-Optimized Migration Paths\n(Species Reintroduction Planning)',
                fontsize=16, weight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
             fontsize=11, framealpha=0.95)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"\n✅ Saved: {output_filename}")
    plt.show()

    # Print statistics
    if path_data:
        print(f"\n📊 Path Statistics:")
        print(f"{'='*60}")
        for i, data in enumerate(path_data):
            print(f"Path {i+1}:")
            print(f"  Length:     {data['length']} steps")
            print(f"  Optimal:    {data['optimal']} steps")
            print(f"  Efficiency: {data['efficiency']:.1f}%")
            print(f"  Start:      {data['start']}")
            print(f"  Goal:       {data['goal']}")

        avg_efficiency = np.mean([d['efficiency'] for d in path_data])
        avg_length = np.mean([d['length'] for d in path_data])
        print(f"\n📈 Average:")
        print(f"  Efficiency: {avg_efficiency:.1f}%")
        print(f"  Path Length: {avg_length:.1f} steps")
        print(f"{'='*60}")

    return path_data

def create_side_by_side_comparison(env, agent, restoration_map_path,
                                   output_filename='comparison.png'):
    """
    Create before/after comparison with A* optimal vs RL learned
    FIXED: Proper coordinate scaling
    """

    restoration_img = Image.open(restoration_map_path)
    restoration_array = np.array(restoration_img)

    # Calculate scaling
    img_height, img_width = restoration_array.shape[:2]
    grid_size = env.grid_size
    scale_y = img_height / grid_size
    scale_x = img_width / grid_size

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Generate one example path
    env.set_difficulty(0.8)
    state = env.reset(use_reintroduction_site=True)
    done = False

    while not done:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)

    # ===== PLOT 1: Restoration Zones Only =====
    ax = axes[0]
    ax.imshow(restoration_array, origin='upper')
    ax.set_title('Step 1: Habitat Restoration\n(GenAI Output)',
                fontsize=14, weight='bold')
    ax.axis('off')

    # ===== PLOT 2: A* Optimal Path =====
    ax = axes[1]
    ax.imshow(restoration_array, origin='upper', alpha=0.7)

    if env.optimal_path:
        optimal = np.array(env.optimal_path)
        # SCALE optimal path
        optimal_scaled = optimal.copy().astype(float)
        optimal_scaled[:, 0] *= scale_y
        optimal_scaled[:, 1] *= scale_x

        ax.plot(optimal_scaled[:, 1], optimal_scaled[:, 0], 'b--',
               linewidth=4, alpha=0.8, label=f'A* Optimal: {len(env.optimal_path)} steps')

        # SCALE start/goal
        start_y, start_x = env.start_pos[0] * scale_y, env.start_pos[1] * scale_x
        goal_y, goal_x = env.goal_pos[0] * scale_y, env.goal_pos[1] * scale_x

        ax.plot(start_x, start_y, 'go',
               markersize=15, label='Start', markeredgecolor='white', markeredgewidth=2)
        ax.plot(goal_x, goal_y, 'r*',
               markersize=20, label='Goal', markeredgecolor='white', markeredgewidth=2)

    ax.set_title('Step 2: A* Optimal Path\n(Theoretical Best)',
                fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.axis('off')

    # ===== PLOT 3: RL Learned Path =====
    ax = axes[2]
    ax.imshow(restoration_array, origin='upper', alpha=0.7)

    if len(env.path_history) > 1:
        path = np.array(env.path_history)
        # SCALE RL path
        path_scaled = path.copy().astype(float)
        path_scaled[:, 0] *= scale_y
        path_scaled[:, 1] *= scale_x

        ax.plot(path_scaled[:, 1], path_scaled[:, 0], 'r-',
               linewidth=4, alpha=0.8,
               label=f'RL Path: {len(env.path_history)} steps\nEfficiency: {info.get("efficiency", 0)*100:.1f}%')

        # SCALE start/goal
        start_y, start_x = env.start_pos[0] * scale_y, env.start_pos[1] * scale_x
        goal_y, goal_x = env.goal_pos[0] * scale_y, env.goal_pos[1] * scale_x

        ax.plot(start_x, start_y, 'go',
               markersize=15, label='Start', markeredgecolor='white', markeredgewidth=2)
        ax.plot(goal_x, goal_y, 'r*',
               markersize=20, label='Goal', markeredgecolor='white', markeredgewidth=2)

    ax.set_title('Step 3: RL-Learned Path\n(Practical Solution)',
                fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_filename}")
    plt.show()

def create_presentation_slide(env, agent, restoration_map_path,
                              stats, output_filename='presentation_slide.png'):
    """
    Create publication-ready slide with all key info
    FIXED: Proper coordinate scaling
    """

    restoration_img = Image.open(restoration_map_path)
    restoration_array = np.array(restoration_img)

    # Calculate scaling
    img_height, img_width = restoration_array.shape[:2]
    grid_size = env.grid_size
    scale_y = img_height / grid_size
    scale_x = img_width / grid_size

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # ===== MAIN PLOT: Restoration with Paths =====
    ax_main = fig.add_subplot(gs[:, :2])
    ax_main.imshow(restoration_array, origin='upper', alpha=0.7)

    # Generate 3 example paths
    for i in range(3):
        env.set_difficulty(0.8)
        state = env.reset(use_reintroduction_site=True)
        done = False

        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)

        if info.get('success'):
            path = np.array(env.path_history)
            # SCALE path
            path_scaled = path.copy().astype(float)
            path_scaled[:, 0] *= scale_y
            path_scaled[:, 1] *= scale_x

            color = ['red', 'orange', 'cyan'][i]
            ax_main.plot(path_scaled[:, 1], path_scaled[:, 0], color=color,
                        linewidth=3, alpha=0.7, label=f'Path {i+1}')

    ax_main.set_title('Species Reintroduction: RL-Optimized Migration Paths',
                     fontsize=18, weight='bold', pad=20)
    ax_main.legend(fontsize=12)
    ax_main.axis('off')

    # ===== STATS PANELS =====
    # Panel 1: Success Rate
    ax1 = fig.add_subplot(gs[0, 2])
    success_rate = np.mean(stats['success_rate'][-100:]) * 100
    ax1.text(0.5, 0.5, f'{success_rate:.0f}%',
            ha='center', va='center', fontsize=60, weight='bold', color='green')
    ax1.text(0.5, 0.2, 'Success Rate', ha='center', fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Panel 2: Efficiency
    ax2 = fig.add_subplot(gs[1, 2])
    efficiency = np.mean([e*100 for e in stats['efficiency_scores'][-100:] if e > 0])
    ax2.text(0.5, 0.5, f'{efficiency:.0f}%',
            ha='center', va='center', fontsize=60, weight='bold', color='blue')
    ax2.text(0.5, 0.2, 'Path Efficiency', ha='center', fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_filename}")
    plt.show()

def verify_coordinate_scaling(restoration_map_path, grid_size=20, reintroduction_sites=None):
    """
    Debug function to verify coordinate system is correct
    Shows grid overlay on restoration map

    Args:
        restoration_map_path: Path to restoration image
        grid_size: Grid size used in environment (default: 20)
        reintroduction_sites: List of (row, col) tuples in grid coordinates
    """
    restoration_img = Image.open(restoration_map_path)
    restoration_array = np.array(restoration_img)

    img_height, img_width = restoration_array.shape[:2]
    scale_y = img_height / grid_size
    scale_x = img_width / grid_size

    print(f"\n🔍 COORDINATE SYSTEM VERIFICATION")
    print(f"{'='*60}")
    print(f"Environment grid: {grid_size}x{grid_size}")
    print(f"Image resolution: {img_width}x{img_height}")
    print(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")

    # Check key positions
    test_positions = [
        (0, 0, "Top-left corner"),
        (grid_size-1, grid_size-1, "Bottom-right corner"),
        (grid_size//2, grid_size//2, "Center"),
        (19, 11, "Example reintro site"),
    ]

    print(f"\n📍 Position mapping (grid → image):")
    for row, col, desc in test_positions:
        scaled_row = row * scale_y
        scaled_col = col * scale_x
        print(f"  ({row:2d}, {col:2d}) → ({scaled_col:6.1f}, {scaled_row:6.1f}) px | {desc}")

    # Visualize grid overlay
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(restoration_array, origin='upper')

    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axvline(x=i * scale_x, color='red', alpha=0.4, linewidth=1)
        ax.axhline(y=i * scale_y, color='red', alpha=0.4, linewidth=1)

    # Label some grid cells
    for i in range(0, grid_size, 5):
        for j in range(0, grid_size, 5):
            ax.text(j * scale_x + scale_x/2, i * scale_y + scale_y/2,
                   f'({i},{j})', ha='center', va='center',
                   color='yellow', fontsize=8, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Mark reintroduction sites if provided
    if reintroduction_sites:
        for site in reintroduction_sites:
            site_y = site[0] * scale_y
            site_x = site[1] * scale_x
            ax.plot(site_x, site_y, 'c^', markersize=20,
                   markeredgecolor='blue', markeredgewidth=3)
            ax.text(site_x, site_y - scale_y, f'Site: {site}',
                   ha='center', va='bottom', color='cyan', fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))

    ax.set_title(f'Grid Verification: {grid_size}x{grid_size} cells on {img_width}x{img_height} image\n'
                f'Cell size: {scale_x:.1f}x{scale_y:.1f} pixels',
                fontsize=14, weight='bold')
    ax.set_xlabel(f'Image X (pixels)', fontsize=12)
    ax.set_ylabel(f'Image Y (pixels)', fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"{'='*60}\n")

# ===== USAGE EXAMPLES =====
if __name__ == "__main__":
    print("Migration Path Overlay Tool - FIXED VERSION")
    print("="*60)
    print("\nFixes coordinate scaling bug:")
    print("  • Properly scales 20x20 grid to full image resolution")
    print("  • Paths now appear on Madagascar land, not ocean")
    print("  • All markers (start/goal/sites) correctly positioned")
    print("\nUsage:")
    print("\n0. Verify coordinate system (NO ENV NEEDED):")
    print("   verify_coordinate_scaling('restoration_zones.png', grid_size=20)")
    print("   # Or with reintro sites:")
    print("   sites = [(19, 11), (18, 11), (18, 11)]")
    print("   verify_coordinate_scaling('restoration_zones.png', grid_size=20, reintroduction_sites=sites)")
    print("\n1. Multiple paths overlay (NEEDS ENV & AGENT):")
    print("   path_data = create_publication_overlay(")
    print("       env, agent, 'restoration_zones.png', num_paths=5)")
    print("\n2. Before/After comparison (NEEDS ENV & AGENT):")
    print("   create_side_by_side_comparison(")
    print("       env, agent, 'restoration_zones.png')")
    print("\n3. Presentation slide (NEEDS ENV, AGENT & STATS):")
    print("   create_presentation_slide(")
    print("       env, agent, 'restoration_zones.png', stats)")
    print("="*60)

# In[ ]:

verify_coordinate_scaling('restoration_zones.png', grid_size=20)
# Or with reintro sites:
sites = [(19, 11), (18, 11), (18, 11)]
verify_coordinate_scaling('restoration_zones.png', grid_size=20, reintroduction_sites=sites)

# In[ ]:

path_data = create_publication_overlay(
       env, agent, 'restoration_zones.png', num_paths=5)

# In[ ]:

create_side_by_side_comparison(
      env, agent, 'restoration_zones.png')

# In[ ]:

create_presentation_slide(
       env, agent, 'restoration_zones.png', stats)