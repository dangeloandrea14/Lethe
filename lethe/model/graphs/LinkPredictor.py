import torch
import torch.nn as nn

from lethe.core.factory_base import get_instance_kvargs


class LinkPredictor(nn.Module):
    """A GNN encoder plus an edge decoder, for the link-prediction arm of Lethe."""

    def __init__(self, encoder, decoder='mlp', decoder_hidden=None, dropout=0.0,
                 n_classes=None):
        super().__init__()

        self.encoder = get_instance_kvargs(encoder['class'], encoder['parameters'])
        self.decoder_type = decoder

        emb_dim = encoder['parameters']['out_channels']
        self.emb_dim = emb_dim
        self.out_channels = emb_dim

        if decoder == 'mlp':
            hidden = decoder_hidden or emb_dim
            self.head = nn.Sequential(
                nn.Linear(2 * emb_dim, hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden, 1),
            )
        elif decoder == 'dot':
            self.head = None
        else:
            raise NotImplementedError(f'Unknown decoder: {decoder}')


    @property
    def hidden_channels(self):
        return self.encoder.hidden_channels

    @property
    def convs(self):
        return self.encoder.convs


    def forward(self, x, edge_index):
        """Node embeddings. Same signature and node-indexed output as the encoders."""
        return self.encoder(x, edge_index)

    def decode(self, z, pairs):
        """Logits for pairs given as ``LongTensor[B, 2]``."""
        pairs = pairs.reshape(-1, 2).to(z.device)
        z_u, z_v = z[pairs[:, 0]], z[pairs[:, 1]]

        if self.decoder_type == 'dot':
            return (z_u * z_v).sum(dim=-1)

        fwd = self.head(torch.cat([z_u, z_v], dim=-1)).squeeze(-1)
        bwd = self.head(torch.cat([z_v, z_u], dim=-1)).squeeze(-1)
        return 0.5 * (fwd + bwd)

    def score_pairs(self, x, edge_index, pairs):
        """Convenience: encode then decode in one call."""
        return self.decode(self(x, edge_index), pairs)
