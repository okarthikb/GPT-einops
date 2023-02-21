import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from math import sqrt


class GELU(nn.Module):
  def forward(self, x):
    return x * 0.5 * (1 + torch.erf(x / sqrt(2)))


class LayerNorm(nn.Module):
  def __init__(self, shape, eps=1e-5, affine=True):
    super().__init__()
    if isinstance(shape, int):
      self.shape = torch.Size([shape])
    elif isinstance(shape, list):
      self.shape = torch.Size(shape)
    assert isinstance(self.shape, torch.Size), 'invalid type for shape'
    self.dims = [-(i + 1) for i in range(len(self.shape))]
    self.eps, self.affine = eps, affine
    if affine:
      self.gamma = nn.Parameter(torch.ones(self.shape))
      self.beta = nn.Parameter(torch.zeros(self.shape))

  def forward(self, x):
    assert x.shape[-len(self.shape):] == self.shape, 'invalid input shape'
    mu = x.mean(dim=self.dims, keepdim=True)  # E[x]
    mu2 = (x * x).mean(dim=self.dims, keepdim=True)  # E[x^2]
    var = mu2 - (mu * mu) + self.eps  # σ^2 = E[x^2] - E[x]^2
    ln = (x - mu) / torch.sqrt(var)
    if self.affine:
      ln = self.gamma * ln + self.beta
    return ln


class Layer(nn.Module):
  def __init__(self, d, nh):
    super().__init__()
    assert d % nh == 0, 'number of heads should divide embedding dim'
    self.d, self.nh, self.h = d, nh, d // nh
    self.wx, self.wo = nn.Linear(d, 3 * d), nn.Linear(d, d)
    self.ln1, self.ln2 = LayerNorm(d), LayerNorm(d)
    self.ffn = nn.Sequential(nn.Linear(d, 4 * d), GELU(), nn.Linear(4 * d, d))

    def fn(m):
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.02)
        nn.init.zeros_(m.bias)

    self.apply(fn)

  def forward(self, xm):
    x, m = xm
    qkv = rearrange(self.wx(self.ln1(x)), 'b l D -> b D l')
    q, k, v = rearrange(qkv, 'b (N h) l -> b N h l', h=self.h).split(self.nh, 1)
    A = F.softmax((einsum('bhri, bhrj -> bhij', q, k) + m) / sqrt(self.d), -1)
    H = einsum('bhic, bhjc -> bhij', v, A)
    MHA = rearrange(rearrange(H, 'b nh h l -> b (nh h) l'), 'b d l -> b l d')
    x = x + self.wo(MHA)
    return x + self.ffn(self.ln2(x)), m


class GPT(nn.Module):
  def __init__(self, d, nh, nl, l, v):
    super().__init__()
    self.l = l
    self.emb = nn.Embedding(v, d)  # token embeddings 
    nn.init.normal_(self.emb.weight, 0, 0.02)
    self.pos = nn.Parameter(torch.randn(l, d) * 0.02)  # learned position embeddings
    m = torch.tril(torch.ones(l, l)) - 1
    m[m == -1] = float('-inf')
    self.m = nn.Parameter(m, requires_grad=False)  # mask
    self.layers = nn.Sequential(*[Layer(d, nh) for _ in range(nl)])  # layers 
    self.out = nn.Linear(d, v, bias=False)  # embedding to logits projection 
    nn.init.normal_(self.out.weight, 0, 0.02)
    self.size = sum(p.numel() for p in self.parameters() if p.requires_grad)

  def forward(self, t):
    l = t.shape[-1]
    assert l <= self.l, f'input sequence length should be <= {self.l}'
    if len(t.shape) == 1:
      t = t.unsqueeze(0)  # (l,) to (1, l)   
    xm = (self.emb(t) + self.pos[:l], self.m[:l, :l])
    return self.out(self.layers(xm)[0]).squeeze()

  def loss(self, it, ot, icl=False):
    ce_loss = F.cross_entropy(
      rearrange(self(it), 'b l v -> (b l) v'),
      rearrange(ot, 'b l -> (b l)'),
      reduction='none'
    )
    if icl:  # compute in-context learning score
      with torch.no_grad():
        icl_score = (ce_loss[-1::self.l] - ce_loss[0::self.l]).mean().item
    return ce_loss.mean(), icl_score
