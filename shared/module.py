import torch
from torch import nn
from torch.nn import functional as F

from .units import CharacterMapper, get_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureActivation(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.activation = nn.Sequential(
            nn.Linear(input_dim, feature_dim // 4, bias=False),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.activation(x)


class SharedWeightLinear(nn.Module):
    def __init__(self, a_dim: int, b_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(b_dim, a_dim))
        nn.init.xavier_uniform_(self.weight)

    def a_to_b(self, a: torch.Tensor):
        # x: [..., a_dim] -> [..., b_dim]
        return F.linear(a, self.weight, bias=None)

    def b_to_a(self, b: torch.Tensor):
        # x: [..., b_dim] -> [..., a_dim]
        return F.linear(b, self.weight.t(), bias=None)


class CharacterEmbedder(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.arange_length = 2**4

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(self.arange_length, n_embd)
        self.position_carry = nn.Linear(n_embd, n_embd, bias=False)

        self.unembeder = nn.Linear(n_embd, vocab_size, bias=False)
        self.unembeder.weight = self.token_embedding_table.weight

    def _positional_embedding(self, T):  # (1, T, C)
        base = self.arange_length

        # collect base-'base' digits (least-significant first), ensure at least one digit
        temp = torch.arange(T, device=device, dtype=torch.long)  # （T,）
        digits = []
        while True:
            digits.append(temp % base)
            temp = temp // base
            if temp.max().item() == 0:
                break

        # process most-significant digit first, carrying via position_carry between digit places
        digits = digits[::-1]
        C = self.position_embedding_table.embedding_dim
        state = torch.zeros((T, C), device=device)
        for idxs in digits:
            state = self.position_carry(state)
            state = state + self.position_embedding_table(idxs)

        # return shape (1, T, C) to broadcast over batch dimension
        return state.unsqueeze(0)

    def _positional_embedding_reverse(self, T):
        pos_emb = self._positional_embedding(T)  # (1, T, C)
        return pos_emb.flip(dims=[1])

    def forward(self, idx):
        return self.embed(idx)

    def embed(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self._positional_embedding_reverse(T)
        x = tok_emb + pos_emb  # (B,T,C)
        return x

    def unembed(self, x):
        return self.unembeder(x)


class AdaptiveVectorModifier(nn.Module):
    def __init__(self, vector_dim: int, modifier_dim: int):
        super().__init__()
        self.vector_dim = vector_dim
        self.modifier_dim = modifier_dim

        self.adaptive_matrix = nn.Sequential(
            nn.Linear(modifier_dim, modifier_dim * 4),
            nn.SiLU(),
            nn.Linear(modifier_dim * 4, modifier_dim * modifier_dim),
        )
        self.vector_map = SharedWeightLinear(vector_dim, modifier_dim)

    def forward(self, x: torch.Tensor):
        pre_shape = x.shape[:-1]
        x = x.view(-1, self.vector_dim)
        # (..., vector_dim) -> (..., feature_dim)
        features = self.vector_map.a_to_b(x)
        # (..., feature_dim) -> (..., feature_dim, feature_dim)
        adaptive_matrix = self.adaptive_matrix(features).view(
            -1, self.modifier_dim, self.modifier_dim
        )
        # (..., feature_dim, feature_dim) @ (..., feature_dim) -> (..., feature_dim)
        features = torch.matmul(adaptive_matrix, features.unsqueeze(-1)).squeeze(-1)
        # (..., feature_dim) -> (..., vector_dim)
        modified_features = self.vector_map.b_to_a(features)
        x = x + modified_features
        return x.view(pre_shape + (self.vector_dim,))


class ForgetModule(nn.Module):
    def __init__(self, vec_dim: int):
        super().__init__()

        self.forget_gate = nn.Sequential(nn.Linear(vec_dim, vec_dim), nn.Tanh())
        self.trans = SharedWeightLinear(vec_dim, vec_dim)

    def forward(self, x: torch.Tensor):
        fr = self.forget_gate(x)
        x = self.trans.a_to_b(x)
        x = x * fr
        x = self.trans.b_to_a(x)
        return x


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril_mask",
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool(),
        )

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # input of size (batch, time_step, channels)
        # output of size (batch, time_step, head_size)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * self.head_size**-0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(
            self.get_buffer("tril_mask")[:T, :T], float("-inf")
        )  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        val = self.value(x)  # (B, T, hs)
        out = wei @ val  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class Head_CompleteInformation(nn.Module):
    def __init__(self, n_embd: int, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x: torch.Tensor):
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * self.head_size**-0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        val = self.value(x)  # (B, T, hs)
        out = wei @ val  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class Head_A2B(nn.Module):
    def __init__(self, a_n_embd: int, b_n_embd: int, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(a_n_embd, head_size, bias=False)
        self.query = nn.Linear(b_n_embd, head_size, bias=False)
        self.value = nn.Linear(a_n_embd, head_size, bias=False)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        k = self.key(a)  # (B, a_size, hs)
        q = self.query(b)  # (B, b_size, hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(
            -2, -1
        )  # (B, b_size, hs) @ (B, hs, a_size) -> (B, b_size, a_size)

        wei = F.softmax(wei * self.head_size**-0.5, dim=-1)  # (B, b_size, a_size)
        # perform the weighted aggregation of the values
        val = self.value(a)  # (B, a_size, hs)
        out = wei @ val  # (B, b_size, a_size) @ (B, a_size, hs) -> (B, b_size, hs)
        return out


class Head_A2B_Lite(nn.Module):
    def __init__(self, a_n_embd: int, b_n_embd: int):
        super().__init__()
        self.a_n_embd = a_n_embd
        self.query = nn.Linear(b_n_embd, a_n_embd, bias=False)

    def forward(self, a, b):
        q = self.query(b)  # (B, b_size, a_n_embd)
        wei = q @ a.transpose(
            -2, -1
        )  # (B, b_size, a_n_embd) @ (B, a_n_embd, a_size) -> (B, b_size, a_size)
        wei = F.softmax(wei * self.a_n_embd**-0.5, dim=-1)  # (B, b_size, a_size)
        out = (
            wei @ a
        )  # (B, b_size, a_size) @ (B, a_size, a_n_embd) -> (B, b_size, a_n_embd)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd, bias=False)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out


class MultiHeadAttention_CompleteInformation(nn.Module):
    def __init__(self, num_heads: int, head_size: int, n_embd: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head_CompleteInformation(n_embd, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd, bias=False)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out


class MultiHeadAttention_A2B(nn.Module):
    def __init__(self, num_heads: int, head_size: int, a_embd: int, b_embd: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head_A2B(a_embd, b_embd, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, b_embd, bias=False)

    def forward(self, a, b):
        out = torch.cat([head(a, b) for head in self.heads], dim=-1)
        out = self.proj(out)  # (B, b_size, b_embd)
        return out


class MultiHeadAttention_A2B_Lite(nn.Module):
    def __init__(self, a_embd: int, b_embd: int):
        super().__init__()
        self.head = Head_A2B_Lite(a_embd, b_embd)
        self.proj = nn.Linear(a_embd, b_embd, bias=False)

    def forward(self, a, b):
        out = self.head(a, b)  # (B, b_size, a_n_embd)
        out = self.proj(out)  # (B, b_size, b_embd)
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd: int, active_nums: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, active_nums),
            nn.SiLU(),
            nn.Linear(active_nums, n_embd, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()

        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=n_embd // n_head,
            n_embd=n_embd,
            block_size=block_size,
        )
        self.ffwd = FeedFoward(n_embd=n_embd, active_nums=4 * n_embd)

    def forward(self, x):
        x = x + self.sa(x)  # 向原特征向量添加修饰
        x = x + self.ffwd(x)  # 从新向量提取信息
        return x


class Block_A2B(nn.Module):
    def __init__(self, a_embd: int, b_embd: int, n_head: int, head_size: int):
        super().__init__()

        self.sa = MultiHeadAttention_A2B(
            num_heads=n_head,
            head_size=head_size,
            a_embd=a_embd,
            b_embd=b_embd,
        )
        self.ffwd = FeedFoward(n_embd=b_embd, active_nums=2 * b_embd)

    def forward(self, a, b):
        b = b + self.sa(a, b)  # 向原特征向量添加修饰
        b = b + self.ffwd(b)  # 从新向量提取信息
        return b


class Block_A2B_Lite(nn.Module):
    def __init__(self, a_embd: int, b_embd: int):
        super().__init__()

        self.sa = MultiHeadAttention_A2B_Lite(
            a_embd=a_embd,
            b_embd=b_embd,
        )
        self.ffwd = FeedFoward(n_embd=b_embd, active_nums=4 * b_embd)

    def forward(self, a, b):
        b = b + self.sa(a, b)  # 向原特征向量添加修饰\
        b = b + self.ffwd(b)  # 从新向量提取信息
        return b


class Block_Self(nn.Module):
    def __init__(self, n_embd: int, n_head: int, head_size: int):
        super().__init__()

        self.self_sa = MultiHeadAttention_CompleteInformation(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
        )
        self.ffwd = FeedFoward(n_embd=n_embd, active_nums=4 * n_embd)

    def forward(self, x):
        added = x + self.self_sa(x)  # 向原特征向量添加修饰
        x = x + self.ffwd(added)  # 从新向量提取信息
        return x


class LLM_ModelBase(nn.Module):
    def __init__(self, vocab_map: CharacterMapper, out_nums: int):
        super().__init__()
        self.iter_n: int = 0

        self.vocab_map: CharacterMapper = vocab_map
        self.out_nums: int = out_nums

    def train_step(
        self,
        data: torch.Tensor,
        max_data_len: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        raise NotImplementedError("train_step method must be implemented in subclass")

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        raise NotImplementedError("generate method must be implemented in subclass")

    def _target_offset_func(self, block_size: int) -> int:
        raise NotImplementedError(
            "target_offset_func method must be implemented in subclass"
        )

    @torch.no_grad()
    def estimate_loss(
        self,
        block_size_range: tuple[int, int],
        batch_size: int,
        eval_iters: int,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
    ):
        out = {}
        self.eval()
        block_size_samples = torch.linspace(
            block_size_range[0],
            block_size_range[1],
            steps=eval_iters,
            dtype=torch.int32,
        ).tolist()
        data = {"train": train_data, "val": val_data}
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(
                    data=data[split],
                    block_size=block_size_samples[k],
                    batch_size=batch_size,
                    target_len=self.out_nums,
                    target_offset=self._target_offset_func(block_size_samples[k]),
                )
                logits, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out
