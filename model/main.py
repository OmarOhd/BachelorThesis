import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import math


# hyperparameters inspired from the paper
N_LAYERS = 8
EMBED_DIM = 256
N_HEADS = 8
D_FF = 256
D_VALUE = 32 # depth of value head, papers default 
D_VALUE_EMBED = 128   # paper: "projection of size 128, which we call the value embedding"

INPUT_ENCODING = [64, 112] # 64 tokens for each square, 112 for leelachess move encoding

PROMOTION_LOGITS = [8, 8, 4] # from, to and piece type promotion logits. 

def _policy_flat(policy_logits, promotion_logits):
    # [B,64,64] + [B,8,8,4] -> [B,4352], order must match encode_move strides
    return torch.cat([policy_logits.flatten(1), promotion_logits.flatten(1)], dim=1)

def compute_loss(policy_logits, promotion_logits, value_logits, policy_targets, value_targets, legal_mask=None):
    # policy_logits:  [B, 64, 64]
    # promotion_logits: [B, 8, 8, 4]
    # value_logits:   [B, 3]
    # policy_targets: [B]   integer in [0, 4096)
    # value_targets:  [B]   integer in [0, 3)
    # returns: scalar tensor
    #flatten and concatenate policy and promotion logits for cross-entropy loss
    # Loss runs outside autocast in the trainer. Upcast before masking and CE:
    # autocast's float32 cross-entropy rule otherwise does not apply here.
    all_moves = _policy_flat(policy_logits, promotion_logits).float() # (batch_size, 4352)
    if legal_mask is not None:
        # Apply the legal move mask to the policy logits
        all_moves = all_moves.masked_fill(~legal_mask, float('-inf'))
    policy_loss = F.cross_entropy(all_moves, policy_targets)
    value_loss = F.cross_entropy(value_logits.float(), value_targets)
    return policy_loss + value_loss


class cf_6M_2024(nn.Module):
    def __init__(self):
        super().__init__()
        
        # first step input projection 
        self.input_proj = nn.Linear(INPUT_ENCODING[1], EMBED_DIM)
        #stack of transformer encoder blocks
        self.blocks = nn.ModuleList([EncoderBlock() for _ in range(N_LAYERS)])
        #policy head
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
    


    def forward(self, board):
        #x shape: (batch_size, board, input_encoding)
        x = self.input_proj(board) # project input to embedding dimension
        for block in self.blocks:
            x = block(x)
        policy_logits, promotion_logits = self.policy_head(x) # (batch_size, 64, 64)
        value_logits = self.value_head(x) # (batch_size, 3)
        return policy_logits, promotion_logits, value_logits




class EncoderBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False) # paper omits bias in QKV projections
        self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        #RMS normalization before attention
        self.norm1 = nn.RMSNorm(EMBED_DIM)
        self.norm2 = nn.RMSNorm(EMBED_DIM)
        #feedforward sublayer
        self.ff_in = nn.Linear(EMBED_DIM, D_FF)
        self.ff_out = nn.Linear(D_FF, EMBED_DIM)
        #split into heads
        self.n_heads = N_HEADS
        self.head_dim = EMBED_DIM // N_HEADS
        #output projection after attention
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        #shaw encoding learnable vectors for relative square positions, index by offset not by pair
        self.a_q = nn.Parameter(torch.zeros(N_HEADS, 15, 15, self.head_dim))
        self.a_k = nn.Parameter(torch.zeros(N_HEADS, 15, 15, self.head_dim))
        self.a_v = nn.Parameter(torch.zeros(N_HEADS, 15, 15, self.head_dim))
        #init with small random values
        nn.init.normal_(self.a_q, std=0.02)
        nn.init.normal_(self.a_k, std=0.02)
        nn.init.normal_(self.a_v, std=0.02)
        # Precompute (dx, dy) offset index for every ordered pair of squares
        i_idx = torch.arange(64)
        rows_i, cols_i = i_idx // 8, i_idx % 8  # (row, col) of square i
        # pairwise offsets: shape [64, 64], values in 0..14
        dx = (rows_i[:, None] - rows_i[None, :]) + 7     # [64, 64]
        dy = (cols_i[:, None] - cols_i[None, :]) + 7     # [64, 64]
        # Register as buffer so it moves with .cuda() / saves with the model
        self.register_buffer("dx", dx)
        self.register_buffer("dy", dy)



    def forward(self, x):
        #attention sublayer with residual connection
        h = self.norm1(x)
        #compute scaled dot-product attention for each head
        q = self.q_proj(h)
        #reshape into (batch_size, board, n_heads, head_dim)
        q = q.reshape(h.size(0), 64, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(h.size(0), 64, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(h.size(0), 64, self.n_heads, self.head_dim).transpose(1, 2)
        # scaled dot-product attention, adding shaw encoding to q k v before softmax
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        # self.a_q is [n_heads, 15, 15, head_dim]; gather to [n_heads, 64, 64, head_dim]
        a_q = self.a_q[:, self.dx, self.dy]   # advanced indexing
        a_k = self.a_k[:, self.dx, self.dy]
        a_v = self.a_v[:, self.dx, self.dy]
        # add shaw encoding contributions to attention weights and output
        attn_weights = attn_weights + torch.einsum("bhid, hijd -> bhij", q, a_k)
        attn_weights = attn_weights + torch.einsum("bhjd, hijd -> bhij", k, a_q)
        attn_weights = attn_weights + torch.einsum("hijd, hijd -> hij", a_q, a_k)
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output + torch.einsum("bhij, hijd -> bhid", attn_weights, a_v)
        # concatenate heads and project
        attn_output = attn_output.transpose(1, 2).reshape(h.size(0), 64, EMBED_DIM)
        attn_out = self.out_proj(attn_output)
        x = x + attn_out
        #feedforward sublayer with residual connection
        h = self.norm2(x)
        h = self.ff_in(h)
        h = F.mish(h) # paper uses mish activation in feedforward
        h = self.ff_out(h)
        x = x + h

        return x

class PolicyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(EMBED_DIM, EMBED_DIM) # Policy embedding
        self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False) # from square query
        self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False) # to square key
        self.promotion_proj = nn.Linear(EMBED_DIM, 4) # promotion piece type logits

    def forward(self, tokens):
        # tokens shape: (batch_size, board, embed_dim) ||-> (batch_size, 64, 256)
        # 1. policy embedding with Mish like paper
        x = self.embed(tokens) # (batch_size, 64, 256)
        x = F.mish(x)
        # 2. Q K projections
        q = self.q_proj(x) # (batch_size, 64, 256)
        k = self.k_proj(x) # (batch_size, 64, 256)
        # 3. compute logits = Q @ K^T / sqrt(EMBED_DIM)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(EMBED_DIM) # (batch_size, 64, 64)
        # 4. promotion logits
        rank8_keys = k[:, 56:64, :] # (batch_size, 8, 256)
        promo_bias = self.promotion_proj(rank8_keys) # (batch_size, 8_to, 4)
        # pull out rank 7 to rank 8 sub-block of the base logits from square 48 to 56, which correspond to moves that could promote
        base_promo = logits[:, 48:56, 56:64] # (batch_size, 8_from, 8_to)
        # broadcast add the promotion bias to the base logits for rank 7 to rank 8 moves
        promotion_logits = base_promo.unsqueeze(-1) + promo_bias.unsqueeze(1) # (batch_size, 8_from, 8_to, 4)
        return logits, promotion_logits

class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_proj = nn.Linear(EMBED_DIM, D_VALUE) # project each token to value head dimension
        self.embed_proj = nn.Linear(64 * D_VALUE, D_VALUE_EMBED) # project flattened token values back to embed dim for final projection
        self.result_proj = nn.Linear(D_VALUE_EMBED, 3) # final projection to 3 logits for win/draw/loss

    def forward(self, tokens):
        # tokens: [B, 64, 256]
        # 1. value projection + Mish
        x = self.value_proj(tokens) # (batch_size, 64, D_VALUE)
        x = F.mish(x)
        # 2. flatten last two dims
        x = x.flatten(1, 2) # (batch_size, 64 * 32)
        # 3. embed projection + Mish
        x = self.embed_proj(x) # (batch_size, 256)
        x = F.mish(x)
        # 4. final projection to 3 logits
        x = self.result_proj(x) # (batch_size, 3)
        return x

if __name__ == "__main__":
    model = cf_6M_2024()
    policy_logits, promotion_logits, value_logits = model(torch.randn(3, 64, 112))
    print(policy_logits.shape)          # torch.Size([3, 64, 64])
    print(value_logits.shape)           # torch.Size([3, 3])
    print(promotion_logits.shape) # torch.Size([3, 8, 8, 4])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} parameters")

    #fake targets for loss test
    fake_board = torch.randn(3, 64, 112)
    policy_targets = torch.randint(0, 64*64 + 8*8*4, (3,))
    value_targets = torch.randint(0, 3, (3,))   

    #optimizer
    optimizer = optim.NAdam(params=model.parameters(), betas=(BETA1, BETA2), eps=EPS, lr=LR)

    #training step
    for step in range(5):
        optimizer.zero_grad()
        policy_logits, promotion_logits, value_logits = model(fake_board)
        loss = compute_loss(policy_logits, promotion_logits, value_logits, policy_targets, value_targets)
        loss.backward() # compute gradients, torch will accumulate them in .grad attributes of parameters
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP) # gradient clipping
        optimizer.step()
        print(f"step {step}: loss = {loss.item():.4f}")
