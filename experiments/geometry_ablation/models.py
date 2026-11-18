"""Independent sequence classifiers; no XAI stroke encoder.

Geometry-5 is (dx, dy, pen_up, centered_x, centered_y), equivalent information
to the team's slide notation (x, y, dx, dy, pen). Tokens remain POINTS.
"""
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SequenceClassifier(nn.Module):
    def __init__(self, family, input_dim, classes, width, depth, heads=4, ff_ratio=3):
        super().__init__()
        self.family = family
        self.input_dim = input_dim
        if width < 1 or depth < 1:
            raise ValueError('Width and depth must be positive')
        if family == 'lstm':
            self.encoder = nn.LSTM(input_dim, width, num_layers=depth,
                                   bidirectional=True, batch_first=True, dropout=0)
            embedding_dim = 2 * width
        elif family == 'transformer':
            if width % heads or width % 2:
                raise ValueError('Width must be even and divisible by heads')
            self.projection = nn.Linear(input_dim, width)
            self.cls = nn.Parameter(torch.empty(1, 1, width))
            nn.init.normal_(self.cls, std=.02)
            # Instantiate independently; avoid cloned layers with identical weights.
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(width, heads, width * ff_ratio, dropout=.1,
                                           activation='gelu', batch_first=True, norm_first=True)
                for _ in range(depth)
            ])
            # Sihui's reference uses zero dropout on attention weights.
            for block in self.blocks:
                block.self_attn.dropout = 0.
            self.norm = nn.LayerNorm(width)
            position = torch.arange(251).float()[:, None]
            frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.) / width))
            pe = torch.zeros(251, width)
            pe[:, 0::2] = torch.sin(position * frequency)
            pe[:, 1::2] = torch.cos(position * frequency)
            self.register_buffer('positions', pe, persistent=False)
            embedding_dim = width
        else:
            raise ValueError(family)
        self.head = nn.Linear(embedding_dim, classes)

    def embed(self, features, lengths):
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError('Wrong feature shape')
        lengths = lengths.to(features.device)
        if torch.any(lengths < 1) or torch.any(lengths > features.size(1)) or features.size(1) > 250:
            raise ValueError('Invalid lengths or more than 250 points')
        if self.family == 'lstm':
            packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.encoder(packed)
            return torch.cat([hidden[-2], hidden[-1]], dim=-1)
        valid = torch.arange(features.size(1), device=features.device)[None, :] < lengths[:, None]
        # Mask before projection too, so arbitrary padding values have no effect.
        x = self.projection(features.masked_fill(~valid[:, :, None], 0))
        x = x + self.positions[1:features.size(1) + 1].to(x.dtype)
        x = torch.cat([self.cls.expand(len(features), -1, -1), x], dim=1)
        valid = torch.cat([torch.ones(len(features), 1, dtype=torch.bool, device=x.device), valid], dim=1)
        for block in self.blocks:
            x = block(x, src_key_padding_mask=~valid)
        return self.norm(x[:, 0])

    def forward(self, features, lengths):
        return self.head(self.embed(features, lengths))


def configurations():
    rows = []
    for family, width, depth in [('lstm', 256, 1), ('transformer', 128, 4)]:
        rows.append(dict(name=f'{family}_point3_control', family=family, input_dim=3,
                         width=width, depth=depth, heads=None if family=='lstm' else 4))
        rows.append(dict(name=f'{family}_geometry5_base', family=family, input_dim=5,
                         width=width, depth=depth, heads=None if family=='lstm' else 4))
        for key, values in [('depth', [2, 3] if family == 'lstm' else [2, 6]),
                            ('width', [128, 512] if family == 'lstm' else [64, 256])]:
            for value in values:
                row = dict(name=f'{family}_geometry5_{key}{value}', family=family,
                           input_dim=5, width=width, depth=depth, heads=None if family=='lstm' else 4)
                row[key] = value
                rows.append(row)
        if family == 'transformer':
            for heads in [2, 8]:
                rows.append(dict(name=f'transformer_geometry5_heads{heads}', family=family,
                                 input_dim=5, width=width, depth=depth, heads=heads))
    primary={'lstm_point3_control':0,'lstm_geometry5_base':1,
             'transformer_point3_control':2,'transformer_geometry5_base':3}
    return sorted(rows,key=lambda row:primary.get(row['name'],4))


def make_model(config, classes):
    return SequenceClassifier(classes=classes, **{k:v for k,v in config.items() if k != 'name'})
