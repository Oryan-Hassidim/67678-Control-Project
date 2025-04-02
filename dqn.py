from collections import deque
from datetime import datetime
from enum import Enum
import random
from typing import Callable
import cv2
import numpy as np
from numpy import ndarray
import pygame
from pygame import Surface
from torch.utils.data import Dataset
import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer, RMSprop


class Action(Enum):
    NOOP = 0
    RIGHT = 1
    LEFT = 2
    JUMP = 3


class Transition:
    def __init__(self, frames: ndarray, action: Action, reward: int, terminal: bool):
        self.frames: ndarray = frames
        self.action: Action = action
        self.reward: int = reward
        self.terminal: bool = terminal

    @property
    def state(self) -> ndarray:
        return self.frames[:-1]

    @property
    def next_state(self) -> ndarray:
        return self.frames[1:]

    def __repr__(self):
        return f"Sample(state={self.state}, action={self.action}, reward={self.reward}, terminal={self.terminal})"

    def __str__(self):
        return f"Sample(state={self.state}, action={self.action}, reward={self.reward}, terminal={self.terminal})"


class ExperienceReplayMemory(Dataset):
    def __init__(self, N: int = 1_000_000):
        super().__init__()
        self.N: int = N
        # D: (N, 84*84+1+1+1) = (N, 84*84 pixels + 1 action + 1 reward + 1 terminal)
        self.D: ndarray = np.zeros((N, 84 * 84 + 1 + 1 + 1), dtype=np.uint8)
        self.D_pointer: int = 0
        self.D_size: int = 0

    def __len__(self) -> int:
        return self.D_size - 1

    def __getitem__(self, idx: int) -> Transition:
        if self.D_size < 4:
            raise IndexError("Empty Replay Memory")

        if not 0 <= idx < self.D_size - 1:
            raise IndexError("Index out of range")

        idx = (self.D_pointer + idx) % self.D_size

        # we want to get the last 4 frames
        indexes = [None, None, None, idx, (idx + 1) % self.D_size]
        repeat, repeat_val = False, idx
        for i in (2, 1, 0):
            if not repeat:
                idx = (idx - 1) % self.D_size
                if self.D_pointer == idx:
                    repeat_val = idx
                    repeat = True
                elif self.D[idx, -1].item():
                    repeat = True
                else:
                    repeat_val = idx
            indexes[i] = repeat_val

        # if indexes are consecutive, we can get regular slice without copying
        if (
            indexes[0]
            == indexes[1] - 1
            == indexes[2] - 2
            == indexes[3] - 3
            == indexes[4] - 4
        ):
            frames = self.D[indexes[0] : indexes[-1] + 1, :-3]
        else:
            frames = self.D[indexes, :-3]

        frames: ndarray = frames.reshape(5, 84, 84)
        action: Action = Action(self.D[idx, -3].item())
        reward: int = self.D[idx, -2].item() - 128
        terminal: bool = bool(self.D[idx, -1].item())
        return Transition(frames, action, reward, terminal)

    def append(
        self, frame: ndarray, action: Action, reward: int, terminal: bool
    ) -> None:
        self.D[self.D_pointer, :-3] = frame.flatten()
        self.D[self.D_pointer, -3] = action.value
        self.D[self.D_pointer, -2] = reward + 128
        self.D[self.D_pointer, -1] = int(terminal)
        self.D_pointer = (self.D_pointer + 1) % self.N
        self.D_size = min(self.D_size + 1, self.N - 1)


class DQNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x: Tensor) -> Tensor:
        # gets: (batch, 4, 84, 84)
        a = nn.functional.relu(self.conv1(x.float() / 255.0))  # (batch, 16, 20, 20)
        b = nn.functional.relu(self.conv2(a))  # (batch, 32, 9, 9)
        c = nn.functional.relu(self.fc1(b.view(-1, 32 * 9 * 9)))  # (batch, 256)
        return self.fc2(c)  # (batch, 4)

    def save_to_file(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_from_file(self, path: str) -> None:
        self.load_state_dict(torch.load(path))


class helpers:
    @staticmethod
    def epsilon(t: int) -> float:
        return 0.1
        return max(0.1, 1 - t / 10_000)


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
            self.replay_memory: ExperienceReplayMemory = ExperienceReplayMemory(N)
            self.batch_size: int = batch_size
            self.epsilon: Callable[[int], float] = epsilon
            self.gamma: float = gamma
            self.total_steps: int = total_steps
        self.dev: str = dev
        self.model: DQNModel = DQNModel().to(dev)
        if not training:
            self.load()
        self.history: deque[ndarray] = deque(maxlen=4)
        self.optimizer: Optimizer = RMSprop(self.model.parameters())
        self.last_action: Action = Action.NOOP
        self.choose_action: Callable[[], Action] = (
            self.choose_action_eps_greedy if training else self.choose_action_best
        )
        self.frame_number: int = 0
        # log_DD.MM.YYYY_HH.MM.SS.txt
        self.log_file = open(f"log_{datetime.now().strftime('%d.%m.%Y_%H.%M.%S')}.txt", "w")

    def phi(self, screen_shot: ndarray) -> ndarray:
        # convert to grayscale using opencv
        gray = cv2.cvtColor(screen_shot, cv2.COLOR_BGR2GRAY)
        # resize to 84x84
        resized = cv2.resize(gray, (84, 84))
        return resized

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
            Qs = self.model(
                torch.stack(
                    [torch.tensor(frame, dtype=torch.float32) for frame in self.history]
                )
                .unsqueeze(0)
                .to(self.dev)
            )
            self.last_action = Action(torch.argmax(Qs).item())
            return self.last_action

    def choose_action_best(self) -> Action:
        frame = self.phi(pygame.surfarray.pixels3d(self.screen))
        self.history.append(frame)
        while len(self.history) < 4:
            self.history.append(frame)

        self.model.eval()
        # Qs = [Q(s, NOOP), Q(s, RIGHT), Q(s, LEFT), Q(s, JUMP)]
        Qs = self.model(
            torch.stack(
                [torch.tensor(frame, dtype=torch.float32) for frame in self.history]
            )
            .unsqueeze(0)
            .to(self.dev)
        )
        self.last_action = Action(torch.argmax(Qs).item())
        return self.last_action

    def update(self, reward: int, terminal: bool) -> None:
        self.replay_memory.append(self.history[-1], self.last_action, reward, terminal)
        if len(self.replay_memory) >= 2 * self.batch_size:
            self.train()

    def train(self) -> None:
        if len(self.replay_memory) < 2 * self.batch_size:
            return  # לא מספיק דוגמאות בזיכרון החוויות

        # Sample random minibatch of transitions (𝜙_𝑗,𝑎_𝑗,𝑟_𝑗,𝜙_(𝑗+1) ) from 𝒟
        # Set 𝑦_𝑗 =
        #   𝑟_𝑗                                   for terminal 𝜙_(𝑗+1)
        #   𝑟_𝑗 + 𝛾 * max┬(𝑎') of 𝑄(𝜙_(𝑗+1),𝑎';𝜃)  for non−terminal 𝜙_(𝑗+1)
        # Perform a gradient descent step on (𝑦_𝑖 − 𝑄(𝜃_𝑖,𝑎_𝑖;𝜃))^2

        # 1. דגום אצווה של דוגמאות מזיכרון החוויות
        indexes = random.sample(range(len(self.replay_memory)), self.batch_size)
        batch = [self.replay_memory[index] for index in indexes]
        batch_states = torch.stack(
            [torch.tensor(s.state, dtype=torch.float32) for s in batch]
        ).to(self.dev)
        batch_next_states = torch.stack(
            [torch.tensor(s.next_state, dtype=torch.float32) for s in batch]
        ).to(self.dev)
        batch_actions = torch.tensor(
            [s.action.value for s in batch], dtype=torch.long, device=self.dev
        )
        batch_rewards = torch.tensor(
            [s.reward for s in batch], dtype=torch.float32, device=self.dev
        )
        batch_terminals = torch.tensor(
            [s.terminal for s in batch], dtype=torch.float32, device=self.dev
        )

        # 2. חישוב Q(s', a') עבור כל s' באצווה
        with torch.no_grad():
            Qs_prime = self.model(batch_next_states).max(1)[0]

        # 3. חישוב Q(s, a) עבור כל s באצווה
        Qs = self.model(batch_states)
        Qs_selected = Qs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

        # 4. חישוב היעד
        targets = batch_rewards + (1 - batch_terminals) * self.gamma * Qs_prime

        # 5. חישוב הפסד
        self.model.train()
        self.optimizer.zero_grad()
        loss = nn.functional.mse_loss(Qs_selected, targets)
        # log the loss to file
        print(f"{self.frame_number},{loss.item()}", file=self.log_file)

        # 6. חישוב גרדיאנטים ועדכון המודל
        loss.backward()
        self.optimizer.step()

    def train_old(self) -> None:
        if len(self.replay_memory) < self.batch_size:
            return  # לא מספיק דוגמאות בזיכרון החוויות

        # 1. דגום אצווה של דוגמאות מזיכרון החוויות
        indexes = random.sample(range(len(self.replay_memory)), self.batch_size)
        batch = [self.replay_memory[index] for index in indexes]
        batch_t = torch.stack(
            [torch.tensor(s.state, dtype=torch.float32) for s in batch]
        ).to(self.dev)

        # 2. חישוב Q(s', a') עבור כל s' באצווה
        with torch.no_grad():
            Qs_prime = self.model(batch_t)

        # 3. חישוב Q(s, a) עבור כל s באצווה
        Qs = torch.zeros(self.batch_size, 4, device=self.dev)
        for i, sample in enumerate(batch):
            Qs[i, sample.action.value] = sample.reward
            if not sample.terminal:
                Qs[i, sample.action.value] += self.gamma * torch.max(Qs_prime[i])

        # 4. חישוב הפסד
        self.model.train()
        self.optimizer.zero_grad()
        loss = nn.functional.mse_loss(self.model(batch_t), Qs)

        # 5. חישוב גרדיאנטים ועדכון המודל
        loss.backward()
        self.optimizer.step()

    def save(self) -> None:
        self.model.save_to_file(self.file_path)
        self.log_file.flush()

    def load(self) -> None:
        self.model.load_from_file(self.file_path)

    def close(self):
        self.log_file.close()
