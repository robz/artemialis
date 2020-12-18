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


def get_notes_on(filename, idxs):
  count = len(idxs)
  notes_on = torch.zeros(count + 1, 128)
  current_notes_on = set()
  for i, idx in enumerate(idxs.to('cpu').numpy()):
    if 0 <= idx < 128: # note-on
      current_notes_on.add(idx)
    elif 128 <= idx < 256: # note-off
      pitch = idx - 128
      if pitch not in current_notes_on:
        print(filename, i, ':', pitch, ' is not on')
      else:
        current_notes_on.remove(idx - 128)
    for note in current_notes_on:
      notes_on[i][note] = 1
  return torch.nonzero(notes_on).int()


onehot = torch.eye(NUM_CHANNELS).to('cuda')
def get_notes_on_input(idxs, notes_on_nonzeros):
  count = len(idxs)
  notes_on = torch.zeros(count + 1, 128, device='cuda')
  #notes_on = torch.zeros(count + 1, 128, device='cpu')
  notes_on_nonzeros = notes_on_nonzeros.long()
  notes_on[notes_on_nonzeros[:, 0], notes_on_nonzeros[:, 1]] = 1
  data_onehot = onehot[idxs]
  return torch.cat([data_onehot, notes_on[:-1]], axis=1)


def augmentIdxs(x, notes_on):
  pitch_transposition = random.randint(-3, 3)
  time_stretch = 0.95 + random.randint(0, 4) * 0.025

  notes_on[:, 1] = torch.clamp(notes_on[:, 1] + pitch_transposition, 0, 127).long()

  note_on_idxs = (x >= 0) * (x < 128)
  x[note_on_idxs] = torch.clamp(x[note_on_idxs] + pitch_transposition, 0, 127).long()

  note_off_idxs = (x >= 128) * (x < 256)
  x[note_off_idxs] = torch.clamp(x[note_off_idxs] + pitch_transposition, 128, 255).long()

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
    os.makedirs(directory + '-tensors-noteons', exist_ok=True)
    print('saving', length, self.phase, 'tracks: ', end=' ')

    total_duration = 0
    total_events = 0
    max_events = 0

    for i in range(length):
      filename = self.df['midi_filename'][i]
      mf = readMidi(directory + '/' + filename)
      x = torch.ShortTensor(midiToIdxs(mf))
      nonzeros = get_notes_on(filename, x)
      os.makedirs(directory + '-tensors/' + os.path.dirname(filename), exist_ok=True)
      torch.save(x, directory + '-tensors/' + filename + '.pt')
      os.makedirs(directory + '-tensors-noteons/' + os.path.dirname(filename), exist_ok=True)
      torch.save(nonzeros, directory + '-tensors-noteons/' + filename + '.pt')

      total_duration += self.df['duration'][i]
      max_events = max(max_events, len(x))
      total_events += len(x)

      if i % 10 == 0:
        print(i, end=' ')

    print()
    print(
      '# of songs:', 
      length, 
      'total duration:', 
      total_duration, 
      '# of events:', 
      total_events, 
      'most events in one song:', 
      max_events
    )

  def fillCache(self, directory=MAESTRO_DIRECTORY):
    print('filling cache for', self.phase)
    printm()
    for i in range(len(self)):
      filename = self.df['midi_filename'][i]
      x = torch.load(directory + '-tensors/' + filename + '.pt')
      notes_on = torch.load(directory + '-tensors-noteons/' + filename + '.pt')
      self.cache[filename] = (x.to('cuda'), notes_on.to('cuda'))
    printm()

  
  """
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
  """

  def __len__(self):
    return self.manual_len if self.manual_len is not None else len(self.df)

  """
  def __getitem__(self, i):
    filename = self.df['midi_filename'][i]
    x = self.cache[filename]
    start = random.randint(0,  max(0, len(x) - self.max_data_len - 1))
    x = x[start:start + self.max_data_len].long()
    augmentIdxs(x)
    return {'input': self.onehot[x[:-1]], 'output': x[1:]}
  """

  def __getitem__(self, i):
    filename = self.df['midi_filename'][i]
    x, notes_on = self.cache[filename]
    start = random.randint(0,  max(0, len(x) - self.max_data_len - 1))
    end = start + self.max_data_len

    x_clip = x[start:end].long()

    notes_on_indices = (notes_on[:, 0] >= start) * (notes_on[:, 0] < end)
    notes_on_clip = notes_on[notes_on_indices].long()
    notes_on_clip[:, 0] -= start

    augmentIdxs(x_clip, notes_on_clip)

    input = get_notes_on_input(x_clip, notes_on_clip)[:-1]
    return {'input': input, 'output': x_clip[1:]}


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
