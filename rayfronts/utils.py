"""Utility functions supporting the other modules"""

import torch

def norm_std(x):
  x = x - x.mean()
  s = x.std()
  if s == 0:
    return torch.ones_like(x)*0.5
  else:
    return x / s

def norm_01(x):
  x = x - x.min()
  mx = x.max()
  if mx == 0:
    return torch.ones_like(x)*0.5
  else:
    return x / mx

def norm_img_01(x):
  B,C,H,W = x.shape
  x = x - torch.min(x.reshape(B,C, H*W), dim=-1).values.reshape(B,C,1,1)
  x = x / torch.max(x.reshape(B,C, H*W), dim=-1).values.reshape(B,C,1,1)
  return x

def compute_cos_sim(vec1: torch.FloatTensor,
                    vec2: torch.FloatTensor,
                    softmax: bool = False) -> torch.FloatTensor:
  """Compute cosine similarity between two batches of D dim vectors.
  
  Args:
    vec1: NxC float tensor representing batch of vectors
    vec2: MxC float tensor representing batch of vectors
    softmax: If False, cosine similarity is returned. If True, softmaxed
      probability is returned across the N dimension.
  Returns:
    result: MxN float tensor representing similarity/prob. where result[0,1]
      represents the similarity of vec1[0] with vec2[1]
  """
  N, C1 = vec1.shape
  M, C2 = vec2.shape
  if C1 != C2:
    raise ValueError(f"vec1 feature dimension '{C1}' does not match vec2"
                     f"feature dimension '{C2}'")
  C = C1

  # Clamp norms to avoid NaNs from zero vectors.
  vec1 = vec1 / vec1.norm(dim = -1, keepdim = True).clamp_min(1e-8)
  vec1 = vec1.reshape(1, N, 1, C)

  vec2 = vec2 / vec2.norm(dim=-1, keepdim = True).clamp_min(1e-8)
  vec2 = vec2.reshape(M, 1, C, 1)

  sim = (vec1 @ vec2).reshape(M, N)
  if softmax:
    return torch.softmax(100 * sim, dim = -1)
  else:
    return sim


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    if len(hex_code) == 3:
        hex_code = ''.join([char * 2 for char in hex_code])
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
