from collections import deque
from enum import Enum
import random
from typing import Callable
import cv2
import numpy as np
from numpy import ndarray
import pygame
from pygame import Surface
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer, RMSprop


class Action(Enum):
    NOOP = 0
    RIGHT = 1
    LEFT = 2
    JUMP = 3


class Sample:
    def __init__(self, state: Tensor, action: Action, reward: int, terminal: bool):
        self.state: Tensor = state
        self.action: Action = action
        self.reward: int = reward
        self.terminal: bool = terminal


class ExperienceReplayMemory(Dataset):
    def __init__(self, N: int = 1_000_000, dev: str = "cpu"):
        super().__init__()
        self.N: int = N
        self.D: Tensor = torch.zeros(
            (N, 84 * 84 + 1 + 1 + 1), dtype=torch.uint8, device=dev
        )
        self.D_pointer: int = 0
        self.D_size: int = 0
        self.dev: str = dev

    def __len__(self) -> int:
        return self.D_size

    def __getitem__(self, idx: int) -> Sample:
        if self.D_size < 3:
            raise IndexError("Empty Replay Memory")

        # we want to get the last 4 frames
        frame_index = idx % self.D_size
        indexes = [None, None, None, frame_index]
        repate, repeat_val = False, frame_index
        for i in range(2, -1, -1):
            if not repate:
                frame_index = (frame_index - 1) % self.D_size
                if self.D[frame_index, -1].item() or self.D_pointer == (frame_index + 1) % self.N:
                    repate = True
                else:
                    repeat_val = frame_index
            indexes[i] = repeat_val

        # if indexes are consecutive, we can get regular slice without copying
        if indexes[0] == indexes[1] - 1 == indexes[2] - 2 == indexes[3] - 3:
            frames = self.D[indexes[0] : indexes[3] + 1, :-3]
        else:
            frames = self.D[indexes, :-3]

        state: Tensor = frames.reshape(4, 84, 84)
        action: Action = Action(self.D[idx, -3].item())
        reward: int = self.D[idx, -2].item() - 128
        terminal: bool = bool(self.D[idx, -1].item())
        return Sample(state, action, reward, terminal)

    def append(
        self, frame: Tensor, action: Action, reward: int, terminal: bool
    ) -> None:
        self.D[self.D_pointer, :-3] = frame.flatten()
        self.D[self.D_pointer, -3] = action.value
        self.D[self.D_pointer, -2] = reward + 128
        self.D[self.D_pointer, -1] = int(terminal)
        self.D_pointer = (self.D_pointer + 1) % self.N
        self.D_size = min(self.D_size + 1, self.N)


class DQNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: implement model
        self.layer1 = nn.Linear(84 * 84 * 4, 4)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement forward pass
        # change to float
        x = x.float()
        x = x.view(-1, 84 * 84 * 4)
        x = self.layer1(x)
        return x

    def save_to_file(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_from_file(self, path: str) -> None:
        self.load_state_dict(torch.load(path))


class helpers:
    @staticmethod
    def epsilon(t: int) -> float:
        return max(0.1, 1 - t / 1_000_000)


class DQN:
    def __init__(
        self,
        screen: Surface,
        file_path: str,
        training: bool = True,
        N: int = 1_000_000,
        total_steps: int = 10_000_000,
        epsilon: Callable[[int], float] = helpers.epsilon,
        gamma: float = 0.9,
        batch_size: int = 32,
        dev: str = "cpu",
    ):
        self.screen: Surface = screen
        self.file_path: str = file_path
        self.training: bool = training
        if training:
            self.replay_memory: ExperienceReplayMemory = ExperienceReplayMemory(N, dev)
            # self.data_loader: DataLoader | None = None
            self.batch_size: int = batch_size
            self.epsilon: Callable[[int], float] = epsilon
            self.gamma: float = gamma
            self.total_steps: int = total_steps
        self.dev: str = dev
        self.model: DQNModel = DQNModel().to(dev)
        if not training:
            self.load()
        self.history: deque = deque(maxlen=4)
        self.optimizer: Optimizer = RMSprop(self.model.parameters())
        self.last_action: Action = Action.NOOP
        self.choose_action: Callable[[], Action] = (
            self.choose_action_eps_greedy if training else self.choose_action_best
        )
        self.frame_number: int = 0

    def phi(self, screen_shot: ndarray) -> Tensor:
        # convert to grayscale using opencv
        gray = cv2.cvtColor(screen_shot, cv2.COLOR_BGR2GRAY)
        # resize to 84x84
        resized = cv2.resize(gray, (84, 84))
        # convert to tensor
        return torch.tensor(resized, dtype=torch.uint8, device=self.dev)

    def reset(self) -> None:
        self.history.clear()

    def choose_action_eps_greedy(self) -> Action:
        self.frame_number += 1
        frame = self.phi(pygame.surfarray.pixels3d(self.screen))
        self.history.append(frame)
        while len(self.history) < 4:
            self.history.append(frame)

        if np.random.rand() < self.epsilon(self.frame_number):
            self.last_action = Action(np.random.randint(4))
            return self.last_action
        else:
            # Qs = [Q(s, NOOP), Q(s, RIGHT), Q(s, LEFT), Q(s, JUMP)]
            self.model.eval()
            Qs = self.model(torch.stack(list(self.history)))
            self.last_action = Action(torch.argmax(Qs).item())
            return self.last_action

    def choose_action_best(self) -> Action:
        frame = self.phi(self.pixel3d)
        self.history.append(frame)
        while len(self.history) < 4:
            self.history.append(frame)

        self.model.eval()
        # Qs = [Q(s, NOOP), Q(s, RIGHT), Q(s, LEFT), Q(s, JUMP)]
        Qs = self.model(torch.stack(list(self.history)))
        self.last_action = Action(torch.argmax(Qs).item())
        return self.last_action

    def update(self, reward: int, terminal: bool) -> None:
        self.replay_memory.append(self.history[-1], self.last_action, reward, terminal)
        if len(self.replay_memory) >= self.batch_size:
            self.train()

    def train(self) -> None:
        # TODO: implement training step
        pass

    def save(self) -> None:
        self.model.save_to_file(self.file_path)

    def load(self) -> None:
        self.model.load_from_file(self.file_path)
