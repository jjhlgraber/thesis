import torch
import torch.nn as nn
import math


class SimpleAbstractorEncoder(nn.Module):
    # __constants__ = ["norm"]

    def __init__(
        self,
        num_layers: int,
        norm=None,
        use_pos_embedding=True,
        use_learned_symbols=True,
        learn_symbol_per_position=False,
        use_symbolic_attention=False,
        symbol_lib_size=8,
        object_dim=64,
        symbol_dim=None,
        dropout=0.1,
        num_heads=1,
        ff_dim=128,
        norm_att=False,
        norm_ff=False,
        resid_att=False,
        resid_ff=False,
        MHA_kwargs={},
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm

        self.use_pos_embedding = use_pos_embedding
        self.use_learned_symbols = use_learned_symbols

        self.learn_symbol_per_position = learn_symbol_per_position
        self.use_symbolic_attention = use_symbolic_attention
        self.symbol_lib_size = symbol_lib_size
        if not (
            self.use_pos_embedding
            or self.use_learned_symbols
            or self.use_symbolic_attention
        ):
            raise Exception(
                "Pick at least one choice for the construction of the symbols"
            )

        self.object_dim = object_dim
        self.symbol_dim = symbol_dim
        if self.symbol_dim is None:
            self.symbol_dim = self.object_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.MHA_kwargs = MHA_kwargs

        encoder_layer = SimpleAbstractorEncoderLayer(
            object_dim=self.object_dim,
            symbol_dim=self.symbol_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            norm_att=norm_att,
            norm_ff=norm_ff,
            resid_att=resid_att,
            resid_ff=resid_ff,
            MHA_kwargs=self.MHA_kwargs,
        )
        self.layers = nn.modules.transformer._get_clones(encoder_layer, num_layers)

        max_seq_len = 3

        if self.use_symbolic_attention:
            self.symbolic_att = SymbolicAttention(
                self.object_dim, self.symbol_dim, self.symbol_lib_size
            )
        else:
            self.initial_symbol_sequence = torch.zeros(
                1, max_seq_len, self.symbol_dim, requires_grad=False
            )

            if self.use_pos_embedding:
                self.add_pos_embedding = AddPositionalEmbedding(
                    embed_dim=self.symbol_dim
                )
                self.initial_symbol_sequence = self.add_pos_embedding(
                    self.initial_symbol_sequence
                )

            if self.use_learned_symbols:
                # if self.learn_symbol_per_position:
                #     learned_symbols_shape = (1, max_seq_len, self.symbol_dim)
                # else:
                #     learned_symbols_shape = self.symbol_dim

                # self.learned_pos_symbols = nn.Parameter(
                #     torch.randn(learned_symbols_shape),
                #     requires_grad=True,
                # )

                self.add_learnable_pos_embedding = AddLearnedPositionalEmbedding(
                    embed_dim=self.symbol_dim, dropout=self.dropout
                )

    def forward(self, X):

        # Algorithm 1: Abstractor module
        # https://arxiv.org/abs/2304.00195
        if self.use_symbolic_attention:
            symbol_seq = self.symbolic_att(X)
        else:
            seq_len = X.size(1)
            symbol_seq = self.initial_symbol_sequence[:, :seq_len].to(device=X.device)
            if self.use_learned_symbols:
                symbol_seq = self.add_learnable_pos_embedding(symbol_seq)

        for mod in self.layers:

            symbol_seq = mod(
                X=X,
                A=symbol_seq,
            )

        if self.norm is not None:
            symbol_seq = self.norm(symbol_seq)

        return symbol_seq


class SimpleAbstractorEncoderLayer(nn.Module):
    def __init__(
        self,
        object_dim,
        symbol_dim,
        num_heads,
        ff_dim,
        dropout=0.1,
        norm_att=False,
        norm_ff=False,
        resid_att=False,
        resid_ff=False,
        MHA_kwargs={},
    ):
        super().__init__()
        self.object_dim = object_dim
        self.symbol_dim = symbol_dim
        self.self_attn = SimpleMultiHeadAttention(
            embed_dim=symbol_dim, kq_dim=object_dim, num_heads=num_heads, **MHA_kwargs
        )
        self.resid_att = resid_att
        self.resid_ff = resid_ff
        if norm_att:
            self.norm1 = nn.LayerNorm(symbol_dim)
        else:
            self.norm1 = nn.Identity()
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(symbol_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, symbol_dim)
        )
        if norm_ff:
            self.norm2 = nn.LayerNorm(symbol_dim)
        else:
            self.norm2 = nn.Identity()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, A, attn_mask=None):
        # Self-attention
        attn_output = self.self_attn(query=X, key=X, value=A, attn_mask=attn_mask)
        if self.resid_att:
            A = A + self.dropout1(attn_output)
        else:
            A = self.dropout1(attn_output)
        A = self.norm1(A)

        # Feedforward network
        ff_output = self.ff(A)
        if self.resid_ff:
            A = A + self.dropout2(ff_output)
        else:
            A = self.dropout2(ff_output)

        A = self.norm2(A)

        return A


class SimpleMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        kq_dim,
        num_heads,
        use_bias=True,
        activation=nn.Softmax(dim=-1),
        use_scaling=True,
        shared_kv_proj=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kq_dim = kq_dim if kq_dim is not None else embed_dim
        self.use_bias = use_bias
        self.activation = activation
        self.use_scaling = use_scaling
        self.shared_kv_proj = shared_kv_proj

        # Linear projections for queries, keys, and values
        self.query_proj = nn.Linear(self.kq_dim, embed_dim, bias=use_bias)
        if shared_kv_proj:
            self.key_proj = self.query_proj
        else:
            self.key_proj = nn.Linear(self.kq_dim, embed_dim, bias=use_bias)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.shape

        # Project queries, keys, and values
        query = self.query_proj(query)
        key = self.key_proj(key)

        value = self.value_proj(value)

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
        #     1, 2
        # )
        # Different since value could be of a more compact shape
        batch_size_value, seq_len_value, _ = value.shape
        value = value.view(
            batch_size_value, seq_len_value, self.num_heads, self.head_dim
        )
        value = value.expand(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        if self.use_scaling:
            scores = scores / (self.head_dim**0.5)

        # Apply attention mask (optional)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        # Apply activation function
        attn_weights = self.activation(scores)

        # with torch.no_grad():
        #     # check symmetry
        #     print(torch.abs(attn_weights - attn_weights.permute(0, 1, 3, 2)).mean())

        # Calculate weighted sum of values
        attn_output = torch.matmul(attn_weights, value)

        # Concatenate and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output


class AddPositionalEmbedding(nn.Module):
    def __init__(self, max_length=3, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.create_positional_encoding()

    def create_positional_encoding(self):
        depth = self.embed_dim / 2

        positions = torch.arange(self.max_length, dtype=torch.float32).unsqueeze(
            1
        )  # (seq, 1)
        depths = (
            torch.arange(depth, dtype=torch.float32).unsqueeze(0) / depth
        )  # (1, depth)
        angle_rates = 1 / (10000**depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)

        self.pos_encoding = torch.cat(
            [torch.sin(angle_rads), torch.cos(angle_rads)], dim=-1
        ).float()

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", self.pos_encoding, persistent=False)

    def forward(self, x):
        if self.pos_encoding is None or self.pos_encoding.shape[-2] < x.shape[-2]:
            self.seq_length, self.embed_dim = x.shape[-2], x.shape[-1]
            self.max_length = max(self.max_length, self.seq_length)
            self.create_positional_encoding()

        seq_length = x.shape[1]

        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= torch.sqrt(
            torch.tensor(self.embed_dim, dtype=torch.float32, device=x.device)
        )
        # add positional encoding
        # hacky, not sure how to fix device issue better
        with torch.no_grad():
            x = x + self.pos_encoding[:seq_length, :].unsqueeze(0).to(x.device)

        return x


class AddLearnedPositionalEmbedding(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        scale: bool = True,
        dropout: float = 0.1,
        max_len: int = 3,
    ):
        super().__init__()
        self.scale = math.sqrt(embed_dim) if scale else 1
        self.position_embeddings = nn.Embedding(max_len, embed_dim)
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        positional_embedding = self.position_embeddings(positions)
        x = self.scale * x + positional_embedding
        return self.dropout(x)


class SymbolicAttention(nn.Module):
    def __init__(self, object_dim, symbol_dim, num_symbols, num_heads=1):
        super().__init__()
        self.object_dim = object_dim
        self.symbol_dim = symbol_dim
        self.num_symbols = num_symbols
        self.num_heads = num_heads

        self.symbol_library = nn.Parameter(
            torch.randn(self.num_symbols, self.symbol_dim),
            requires_grad=True,
        )
        self.binding_vectors = nn.Parameter(
            torch.randn(self.num_symbols, self.object_dim),
            requires_grad=True,
        )
        self.query_proj = nn.Linear(self.object_dim, self.object_dim * self.num_heads)

    def forward(self, X):
        batch_size, seq_len, _ = X.shape

        # Project queries for multi-head attention
        query = self.query_proj(X).view(
            batch_size, seq_len, self.num_heads, self.object_dim
        )
        query = query.transpose(1, 2)  # (batch_size, num_heads, seq_len, object_dim)

        # Calculate attention scores
        scores = torch.matmul(
            query, self.binding_vectors.T
        )  # (batch_size, num_heads, seq_len, num_symbols)

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(
            scores, dim=-1
        )  # (batch_size, num_heads, seq_len, num_symbols)

        # Retrieve symbols using attention weights
        retrieved_symbols = torch.matmul(
            attn_weights, self.symbol_library
        )  # (batch_size, num_heads, seq_len, symbol_dim)

        # Concatenate multi-head outputs
        retrieved_symbols = (
            retrieved_symbols.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )

        return retrieved_symbols
