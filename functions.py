from IPython.display import display, SVG
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
from peewee import *
import base64
import torch
from torch import nn
from torch.utils.data import IterableDataset
from random import randint

# lower case is black, upper case is white
PIECES = {"r": 0,
          "n": 1,
          "b": 2,
          "q": 3,
          "k": 4,
          "p": 5,
          "R": 6,
          "N": 7,
          "B": 8,
          "Q": 9,
          "K": 10,
          "P": 11,
          }
PIECES_NAME = {"r": "black rook",
               "n": "black knight",
               "b": "black bishop",
               "q": "black queen",
               "k": "black king",
               "p": "black pawn",
               "R": "white rook",
               "N": "white knight",
               "B": "white bishop",
               "Q": "white queen",
               "K": "white king",
               "P": "white pawn"
               }
SVG_BASE_URL = "https://us-central1-spearsx.cloudfunctions.net/chesspic-fen-image/"
db = SqliteDatabase('evaluations.db')
db.connect()

class Evaluations(Model):
  id = IntegerField()
  fen = TextField()
  binary = BlobField()
  eval = FloatField()

  class Meta:
    database = db

  def binary_base64(self):
    return base64.b64encode(self.binary)

class EvaluationDataset(IterableDataset):
  def __init__(self, count):
    self.count = count
  def __iter__(self):
    return self
  def __next__(self):
    idx = randint(1, self.count)
    return self[idx]
  def __len__(self):
    return self.count
  def __getitem__(self, idx):
    eval = Evaluations.get(Evaluations.id == idx)
    fen = fen_preprocess(eval.fen).astype(np.single)
    score = max(eval.eval, -15)
    score = min(score, 15)
    score = np.array([score]).astype(np.single)
    return {'id': idx, 'fen':fen, 'score':score}

def fen_preprocess(fen):
  VEC_LEN = 768
  vec = np.array([0]*VEC_LEN)
  fen = fen.split()[0]
  fen = re.sub(r'\d', lambda x: '0' * int(x.group()), fen)
  fen = fen.replace("/", "")
  for i in range(len(fen)):
    if fen[i] != "0" :
      position = i * 12 + PIECES[fen[i]]
      vec[position] = 1
  return vec

def train(model, config, data_loader, device, n_epoch, track=False):
  l1_weight_tracker = []
  l1_bias_tracker = []
  loss = nn.MSELoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


  n_total_steps = len(data_loader)
  for epoch in range(n_epoch):
    for i, batch in enumerate(data_loader):
      fens = batch["fen"].to(device)
      scores = batch["score"].to(device)

      outputs = model(fens)
      l = loss(outputs, scores)

      l.backward()
      optimizer.step()
      optimizer.zero_grad()

      if i % 20 == 0:
        if track:
          param = model.state_dict()
          l1_weight_tracker.append(np.array(param["l1.weight"].detach().clone().cpu()))
          l1_bias_tracker.append(np.array(param["l1.bias"].detach().clone().cpu()))
        print("Epoch [{}/{}], Step [{}/{}], Loss: {}".format(epoch+1, n_epoch, i, n_total_steps, l))
  return l1_weight_tracker, l1_bias_tracker


'''interpreting functions'''
def get_piece_weight(arr, piece):
  return np.array(arr[:, [i + PIECES[piece] for i in range(0, 768, 12)]])

def arrange_to_piece(arr):
  piece_arr = np.zeros(arr.shape)
  for piece, number in PIECES.items():
    piece_arr[:, [number * 64 + i for i in range(64)]] = get_piece_weight(arr, piece)
  return piece_arr

def get_row_abs_mean(weight):
  abs_mean = []
  for i in range(weight.shape[0]):
    abs_mean.append(np.mean(np.absolute(weight[i, :])))
  return abs_mean

def scale_abs_mean(abs_mean: dict):
  multiplier = 1 / np.mean(abs_mean["p"])
  for key in abs_mean.keys():
    abs_mean[key] = np.array(abs_mean[key]) * multiplier
  return abs_mean

def plot_abs_mean(abs_mean):
  fig = plt.figure(figsize=(15,10))
  max_val = np.max(np.array(list(abs_mean.values())))
  for piece, avg in abs_mean.items():
    plt.plot(range(0, len(avg)), avg, label=piece)
  plt.ylim((0, max_val + 2))
  plt.axhline(y=9, linewidth=1, linestyle="dashed")
  plt.axhline(y=5, linewidth=1, linestyle="dashed")
  plt.axhline(y=3, linewidth=1, linestyle="dashed")
  plt.axhline(y=1, linewidth=1, linestyle="dashed")
  plt.legend(loc=1)
  plt.show()

def heatmap_weight(arr, piece):
  if arr.shape[1] != 8 * 8:
    raise Exception("Not chess board dimension!")

  vmin = np.min(arr)
  vmax = np.max(arr)

  fig, ax = plt.subplots(1, arr.shape[0], figsize=(80, 10))
  fig.suptitle("Piece: {}".format(PIECES_NAME[piece]), fontsize=50)
  for i in range(arr.shape[0]):
    weight = arr[i, :].reshape(8, 8)
    sns.heatmap(data=weight,
                vmax=vmax,
                vmin=vmin,
                ax=ax[i],
                annot=True,
                cmap='RdYlGn',
                linewidths=0.5)
    ax[i].set_title("row {}".format(i + 1), fontsize=20)
  plt.show()

def svg_url(fen):
  fen_board = fen.split()[0]
  return SVG_BASE_URL + fen_board

def display_svg(fen):
  display(SVG(svg_url(fen)))

def gs(X):
  Q, R = torch.linalg.qr(X)
  return Q

def change_of_basis(coeff, original_basis, new_basis):
  assert original_basis.shape[0] == new_basis.shape[0] and \
    new_basis.shape[0] == coeff.shape[1], "basis vectors must have same dimension"
  assert original_basis.shape[1] == new_basis.shape[1] and \
    new_basis.shape[1] == coeff.shape[0], "must have same number of basis vector"
  return torch.linalg.lstsq(new_basis, torch.matmul(original_basis, coeff)).solution