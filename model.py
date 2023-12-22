from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class ModelConfig:
  n_layers = 3
  input_dim = 768  # 64 * 6 * 2
  l1_output_dim = 8
  output_dim = 1
  batch_size = 256
  learning_rate = 1e-3
  data_count = 37164639

class EvalModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.batch_size = config.batch_size
    self.learning_rate = config.learning_rate
    self.l1 = nn.Linear(config.input_dim, config.l1_output_dim)
    self.l2 = nn.Linear(config.l1_output_dim, config.output_dim)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.l1(x)
    x = self.relu(x)
    x = self.l2(x)
    return x