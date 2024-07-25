import torch
import random
from copy import deepcopy


class LatentCodePool:
    def __init__(self, size, emotions, actors):
        self.size = size
        self.emotions = emotions
        self.pool = {}
        for actor in actors:
            self.pool[actor] = {}
            for emotion in emotions:
                self.pool[actor][emotion] = []

    def query(self, emotion, actors):
        ws = []
        w_pos = random.choice(self.pool[actors][emotion])
        ws.append(w_pos)
        for emo in self.emotions:
            if emo == emotion:
                continue

            w_neg = random.choice(self.pool[actors][emo])
            ws.append(w_neg)

        ws = torch.stack(ws, dim=0).cuda()

        return ws

    def add_latent(self, emotions, actors, ws):
        ws = ws.detach().cpu()
        for i in range(ws.shape[0]):
            a = actors[i]
            e = emotions[i]
            w = ws[i]

            if len(self.pool[a][e]) >= self.size:
                self.pool[a][e].pop(0)
            self.pool[a][e].append(w)

    def load(self, pool):
        self.pool = deepcopy(pool)
