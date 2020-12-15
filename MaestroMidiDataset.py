import random, time, os
import pandas as pd
import torch
import platform

from GPUUsageUtils import printm
from MidiIndexUtils import NUM_CHANNELS, readMidi, midiToIdxs

MAESTRO_DIRECTORY = 'maestro-v2.0.0' if platform.system() == 'Linux' else 'Music/maestro-v2.0.0'

####################################################################
# Dataset class, dataloader, data augmentation
####################################################################


def augmentIdxs(x):
  pitch_transposition = random.randint(-3, 3)
  time_stretch = 0.95 + random.randint(0, 4) * 0.025

  note_ons = (x >= 0) * (x < 128)
  x[note_ons] = torch.clamp(x[note_ons] + pitch_transposition, 0, 127).long()

  note_offs = (x >= 128) * (x < 256)
  x[note_offs] = torch.clamp(x[note_offs] + pitch_transposition, 128, 255).long()

  time_shifts = (x >= 256) * (x < 356)
  x[time_shifts] = torch.clamp((x[time_shifts] - 256) * time_stretch, 0, 99).long() + 256

  return pitch_transposition, time_stretch


class MaestroDataset(torch.utils.data.Dataset):
  # phase is train or eval
  def __init__(self, phase, directory=MAESTRO_DIRECTORY, channels=NUM_CHANNELS, len=None, max_data_len=10000):
    df = pd.read_csv(directory + '/maestro-v2.0.0.csv')
    self.df = df[(df['split'] == 'train') if phase == 'train' else (df['split'] != 'train')]
    self.df.reset_index(drop=True, inplace=True)
    self.onehot = torch.eye(channels).to('cuda')
    self.phase = phase
    self.manual_len = len
    self.max_data_len = max_data_len
    self.cache = dict()

  def preprocessAndSaveToDisk(self, directory=MAESTRO_DIRECTORY):
    length = len(self)
    os.makedirs(directory + '-tensors', exist_ok=True)
    for i in range(length):
      filename = self.df['midi_filename'][i]
      mf = readMidi(directory + '/' + filename)
      x = torch.ShortTensor(midiToIdxs(mf))
      os.makedirs(directory + '-tensors/' + os.path.dirname(filename), exist_ok=True)
      torch.save(x, directory + '-tensors/' + filename + '.pt')
      if i % 10 == 0:
        print(self.phase, i + 1, '/', length)
  
  def fillCache(self, directory=MAESTRO_DIRECTORY):
    print('filling cache for', self.phase)
    printm()
    for i in range(len(self)):
      filename = self.df['midi_filename'][i]
      self.cache[filename] = torch.load(directory + '-tensors/' + filename + '.pt').to('cuda')
    printm()

  def __len__(self):
    return self.manual_len if self.manual_len is not None else len(self.df)

  def __getitem__(self, i):
    filename = self.df['midi_filename'][i]
    x = self.cache[filename]
    start = random.randint(0,  max(0, len(x) - self.max_data_len - 1))
    x = x[start:start + self.max_data_len].long()
    augmentIdxs(x)
    return {'input': self.onehot[x[:-1]], 'output': x[1:]}


def my_collate(data):
  min_len = min([e['output'].shape[0] for e in data])
  return {
    'output': torch.stack([e['output'][:min_len] for e in data]), 
    'input': torch.stack([e['input'][:min_len, :] for e in data]),
  }


def get_dataloader(phase, len=None, max_data_len=10000):
  dataset = MaestroDataset(phase, len=len, max_data_len=max_data_len)
  dataset.fillCache()
  return torch.utils.data.DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=0, 
    collate_fn=my_collate
  )


def preprocessAndSaveToDisk():
  for phase in ['eval', 'train']:
    dataset = MaestroDataset(phase)
    start = time.time()
    dataset.preprocessAndSaveToDisk()
    print('delta time:', time.time() - start)


if __name__ == "__main__":
  preprocessAndSaveToDisk()
