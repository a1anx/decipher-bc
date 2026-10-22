from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class ConditionalDenseNN(torch.nn.Module):
    """Dense neural network with multiple outputs, optionally conditioned on a context variable.

    (Derived from pyro.nn.dense_nn.ConditionalDenseNN with some modifications [1])

    Parameters
    ----------
    input_dim : int
        Dimension of the input
    hidden_dims : sequence of ints
        Dimensions of the hidden layers (excluding the output layer)
    output_dims : sequence of ints (optional)
        Dimensions of each output layer
        Default: (1,)
    context_dim : int (optional)
        Dimension of the context input.
        Default: 0. No context input.
    deep_context_injection : bool (optional)
        If True, inject the context into every hidden layer.
        If False, only inject the context into the first hidden layer (concatenated with the input).
        Default: False.
    activation : torch.nn.Module (optional)
        Activation function to use between hidden layers (not applied to the outputs).
        Default: torch.nn.ReLU()
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dims: Sequence = (1,),
        context_dim: int = 0,
        deep_context_injection: bool = False,
        activation=torch.nn.ReLU(),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.deep_context_injection = deep_context_injection
        self.n_output_layers = len(self.output_dims)
        self.output_total_dim = sum(self.output_dims)

        # The multiple outputs are computed as a single output layer, and then split
        indices = np.concatenate(([0], np.cumsum(self.output_dims)))
        self.output_slices = [slice(s, e) for s, e in zip(indices[:-1], indices[1:])]

        # Create masked layers
        deep_context_dim = self.context_dim if self.deep_context_injection else 0
        layers = []
        batch_norms = []
        if len(hidden_dims):
            layers.append(torch.nn.Linear(input_dim + context_dim, hidden_dims[0]))
            batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
            for i in range(1, len(hidden_dims)):
                layers.append(
                    torch.nn.Linear(hidden_dims[i - 1] + deep_context_dim, hidden_dims[i])
                )
                batch_norms.append(nn.BatchNorm1d(hidden_dims[i]))

            layers.append(
                torch.nn.Linear(hidden_dims[-1] + deep_context_dim, self.output_total_dim)
            )
        else:
            layers.append(torch.nn.Linear(input_dim + context_dim, self.output_total_dim))

        self.layers = torch.nn.ModuleList(layers)

        self.f = activation
        self.batch_norms = torch.nn.ModuleList(batch_norms)

    def forward(self, x, context=None):
        if context is not None:
            # We must be able to broadcast the size of the context over the input
            context = context.expand(x.size()[:-1] + (context.size(-1),))

        h = x
        for i, layer in enumerate(self.layers):
            if self.context_dim > 0 and (self.deep_context_injection or i == 0):
                h = torch.cat([context, h], dim=-1)
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.batch_norms[i](h)
                h = self.f(h)

        if self.n_output_layers == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_total_dim])

            if self.n_output_layers == 1:
                return h

            else:
                return tuple([h[..., s] for s in self.output_slices])

#EDIT: AN ENTIRELY NEW CLASS HAS BEEN ADDED
class BatchCorrectedDecoder(nn.Module):
    """Decoder where z attends over all n_batches prototype embeddings via cross-attention.

    key/value: all n_batches prototypes, shape (n_batches, batch_size, batch_emb_dim)
    query:     cell's projected z,       shape (1,         batch_size, batch_emb_dim)

    Softmax is over n_batches keys, so z actually drives the attention weights.
    """

    def __init__(
        self,
        latent_dim: int,
        n_output: int,
        n_batches: int,
        batch_emb_dim: int = 64,
        hidden_dims: Sequence[int] = (),
        n_heads: int = 4,
        dropout: float = 0.1,
        combination_mode: str = "concat",
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_output = n_output
        self.n_batches = n_batches
        self.batch_emb_dim = batch_emb_dim
        self.combination_mode = combination_mode
        self.use_layer_norm = use_layer_norm

        self.batch_embedding = nn.Embedding(n_batches, batch_emb_dim)

        if latent_dim != batch_emb_dim:
            self.z_projection = nn.Linear(latent_dim, batch_emb_dim)
        else:
            self.z_projection = nn.Identity()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=batch_emb_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False,
        )

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(batch_emb_dim)

        if combination_mode == "concat":
            mlp_input_dim = latent_dim + batch_emb_dim
        elif combination_mode == "add":
            if latent_dim != batch_emb_dim:
                raise ValueError(
                    f"'add' mode requires latent_dim == batch_emb_dim, "
                    f"got {latent_dim} vs {batch_emb_dim}"
                )
            mlp_input_dim = latent_dim
        else:
            raise ValueError(f"Unknown combination_mode: {combination_mode}")

        layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_output))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, z, batch_index):
        """
        Args:
            z:           (batch_size, latent_dim)
            batch_index: (batch_size,) integer indices — unused in attention path,
                         kept for API consistency with the calling code.
        Returns:
            logits: (batch_size, n_output)
        """
        batch_size = z.shape[0]

        z_proj = self.z_projection(z)                                       # (batch_size, batch_emb_dim)

        all_batch_embs = self.batch_embedding.weight                        # (n_batches, batch_emb_dim)
        query = z_proj.unsqueeze(0)                                         # (1, batch_size, batch_emb_dim)
        key   = all_batch_embs.unsqueeze(1).expand(-1, batch_size, -1)     # (n_batches, batch_size, batch_emb_dim)
        value = key

        attn_output, _ = self.multihead_attn(query=query, key=key, value=value)
        attn_output = attn_output.squeeze(0)                                # (batch_size, batch_emb_dim)

        if self.use_layer_norm:
            attn_output = self.layer_norm(attn_output)

        if self.combination_mode == "concat":
            combined = torch.cat([z, attn_output], dim=-1)
        else:  # "add"
            combined = z + attn_output

        return self.mlp(combined)                                           # (batch_size, n_output)

    def get_attention_weights(self, z):
        """
        Returns:
            attn_weights: (batch_size, n_heads, 1, n_batches)
        """
        batch_size = z.shape[0]

        z_proj = self.z_projection(z)
        all_batch_embs = self.batch_embedding.weight
        query = z_proj.unsqueeze(0)
        key   = all_batch_embs.unsqueeze(1).expand(-1, batch_size, -1)
        value = key

        _, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            need_weights=True,
            average_attn_weights=False,
        )
        return attn_weights                                                  # (batch_size, n_heads, 1, n_batches)
