import cv2
import torch
from dqn import Sample, ExperienceReplayMemory
import time

replay_memory = ExperienceReplayMemory(N=1_000)
replay_memory.D = torch.load("replay_memory.pkl")
replay_memory.D_pointer = 600
replay_memory.D_size = 1000


# for i in range(500):
#     mem = replay_memory.D[i]
#     frame = mem[:-3].view(84, 84).numpy()
#     # resize the image to 4x
#     frame = cv2.resize(frame, (84 * 4, 84 * 4), interpolation=cv2.INTER_NEAREST)
#     # add text to the image
#     cv2.putText(
#         frame,
#         f"{i:>3} {mem[-2] + 128:>2} {'terminal' if mem[-1] else ''}",
#         (0, 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (255, 255, 255),
#         2,
#     )
#     # cv2.imshow(f"{i:>3} {mem[-2]} {"terminal" if mem[-1] else ''}", frame)
#     cv2.imshow("frame", frame)
#     cv2.waitKey(300 if mem[-1] else 100)

# get the first terminal frame
# terminal_idx = 0
# while not replay_memory.D[terminal_idx, -1]:
#     terminal_idx += 1

replay_memory.D_pointer = 60

sample = replay_memory[59]
sample = replay_memory[60]
sample = replay_memory[61]


