from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class EpisodeMetrics:
    episode_length: int
    episode_return: float


@dataclass
class Episode:
    obs: torch.ByteTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.LongTensor
    trunc: torch.ByteTensor
    mask_padding: torch.BoolTensor = field(init=False)

    def __post_init__(self):
        # assert len(self.obs) == len(self.act) == len(self.rew) == len(self.end) == len(self.mask_padding)
        self.mask_padding = torch.ones_like(self.trunc, dtype=torch.bool)
        assert len(self.obs) == len(self.act) == len(self.rew) == len(self.end) == len(self.trunc) == len(self.mask_padding)
        
        self.end = self.end.to(torch.long)    # TODO: be careful about overflows with masked_fill !!

        if self.end.sum() > 0:
            idx_end = torch.argmax(self.end) + 1
            self.obs = self.obs[:idx_end]
            self.act = self.act[:idx_end]
            self.rew = self.rew[:idx_end]
            self.end = self.end[:idx_end]
            self.trunc = self.trunc[:idx_end]
            self.mask_padding = torch.ones(idx_end, dtype=torch.bool) # self.mask_padding[:idx_end]


    def __len__(self) -> int:
        return self.obs.size(0)

    def merge(self, other: Episode) -> Episode:
        return Episode(
            torch.cat((self.obs, other.obs), dim=0),
            torch.cat((self.act, other.act), dim=0),
            torch.cat((self.rew, other.rew), dim=0),
            torch.cat((self.end, other.end), dim=0),
            torch.cat((self.trunc, other.trunc), dim=0),
            torch.cat((self.mask_padding, other.mask_padding), dim=0),
        )

    def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
        assert start < len(self) and stop > 0 and start < stop
        padding_length_right = max(0, stop - len(self))
        padding_length_left = max(0, -start)
        assert padding_length_right == padding_length_left == 0 or should_pad

        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(len(self), stop)
        segment = Episode(
            obs=self.obs[start:stop],
            act=self.act[start:stop],
            rew=self.rew[start:stop],
            end=self.end[start:stop],
            trunc=self.trunc[start:stop],
            # self.mask_padding[start:stop],
        )

        segment.obs = pad(segment.obs)
        segment.act = pad(segment.act)
        segment.rew = pad(segment.rew)
        segment.end = pad(segment.end)
        segment.trunc = pad(segment.trunc)
        segment.mask_padding = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.mask_padding, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)

        return segment

    def compute_metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(len(self), self.rew.sum())

    def save(self, path: Path) -> None:
        torch.save(self.__dict__, path)
