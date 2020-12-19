import os
import torch

from GPUUsageUtils import printm
from MidiIndexUtils import NUM_CHANNELS
from MaestroMidiDataset import MaestroMidiDataset, MAESTRO_DIRECTORY


####################################################################
# Dataset class for 128-element note on vector conditioning
####################################################################


def get_notes_on_sparse(filename, idxs):
  notes_on = torch.zeros(len(idxs), 128)
  current_notes_on = set()
  for i, idx in enumerate(idxs.numpy()):
    if 0 <= idx < 128: # note-on
      current_notes_on.add(idx)
    elif 128 <= idx < 256: # note-off
      pitch = idx - 128
      if pitch not in current_notes_on:
        print(filename, i, ':', pitch, ' is not on')
      else:
        current_notes_on.remove(pitch)
    for note in current_notes_on:
      notes_on[i, note] = 1
  # most notes are off most of the time, so store them sparsely as indicies
  return torch.nonzero(notes_on).int()


class MaestroMidiDatasetWithConditioning(MaestroMidiDataset):
  # phase is train or eval
  def __init__(self, phase, directory=MAESTRO_DIRECTORY, channels=NUM_CHANNELS, len=None, max_data_len=10000):
    super().__init__(phase=phase, directory=directory, channels=channels, len=len, max_data_len=max_data_len)
    self.sparse_conditioning_cache = dict()

  def preprocessAndSaveToDisk(self, directory=MAESTRO_DIRECTORY):
    super().preprocessAndSaveToDisk(directory=directory)
    super().fillCache(directory=directory)
    os.makedirs(directory + '-tensors-noteons', exist_ok=True)
    for i in range(len(self)):
      filename = self.df['midi_filename'][i]
      x = self.cache[filename]
      notes_on_sparse = get_notes_on_sparse(filename, x.to('cpu'))
      os.makedirs(directory + '-tensors-noteons/' + os.path.dirname(filename), exist_ok=True)
      torch.save(notes_on_sparse, directory + '-tensors-noteons/' + filename + '.pt')
      if i % 10 == 0:
        print(i, end=' ')

  def fillCache(self, directory=MAESTRO_DIRECTORY):
    super().fillCache(directory=directory)
    print('filling conditioning cache for', self.phase)
    printm()
    for i in range(len(self)):
      filename = self.df['midi_filename'][i]
      notes_on_sparse = torch.load(directory + '-tensors-noteons/' + filename + '.pt')
      self.sparse_conditioning_cache[filename] = notes_on_sparse.to('cuda')
    printm()

  def getFullItem(self, i, seq_len):
    item = super().getFullItem(i, seq_len)
    idxs, input = [item[k] for k in ['idxs', 'input']]
    seq_len = len(idxs)

    filename = self.df['midi_filename'][i]
    notes_on_sparse = self.sparse_conditioning_cache[filename]
    clip_indices = notes_on_sparse[:, 0] < seq_len
    notes_on_sparse_clip = notes_on_sparse[clip_indices].long()

    notes_on = torch.zeros(seq_len, 128, device='cuda')
    notes_on[notes_on_sparse_clip[:, 0], notes_on_sparse_clip[:, 1]] = 1

    return {**item, 'input': torch.cat([input, notes_on], axis=1)}

  def __getitem__(self, i):
    item = super().__getitem__(i)
    input, start, pitch_transposition = [item[k] for k in ['input', 'start', 'pitch_transposition']]
    seq_len = len(input)

    filename = self.df['midi_filename'][i]
    notes_on_sparse = self.sparse_conditioning_cache[filename]

    # apply data augmentation: random clipping + pitch transposition
    clip_indices = (notes_on_sparse[:, 0] >= start) * (notes_on_sparse[:, 0] < start + seq_len)
    notes_on_sparse_clip = notes_on_sparse[clip_indices].long()
    notes_on_sparse_clip[:, 0] -= start
    notes_on_sparse_clip[:, 1] = torch.clamp(notes_on_sparse_clip[:, 1] + pitch_transposition, 0, 127).long()

    # un-sparsify the conditioning matrix
    notes_on = torch.zeros(seq_len, 128, device='cuda')
    notes_on[notes_on_sparse_clip[:, 0], notes_on_sparse_clip[:, 1]] = 1

    # concat it to the end of the one-hot input events
    return {**item, 'input': torch.cat([input, notes_on], axis=1)}


transition_matrix = torch.cat([
  torch.diag(torch.ones(128)),
  torch.diag(torch.empty(128).fill_(-1)),
  torch.zeros(128, 388-256)
], axis=1).to('cuda')

def get_condition(prime_condition=None, record_inputs=True):
  # compute current conditioning from prime
  if prime_condition is not None:
    #current_notes_on = [prime[0, -1, -128:]]
    current_notes_on = [prime_condition]
  else:
    current_notes_on = [torch.zeros(128, device='cuda')]

  if record_inputs:
    inputs = []
  else:
    inputs = None

  def condition(event):
    # update current conditioning based on latest event
    current_notes_on[0] = torch.clamp(current_notes_on[0] + torch.matmul(transition_matrix, event), 0, 1)
    if record_inputs:
      inputs.append(current_notes_on[0])
    # concat it to the input and return it
    input = torch.cat([event, current_notes_on[0]])
    return input

  return condition, inputs


if __name__ == "__main__":
  MaestroMidiDatasetWithConditioning.preprocessAllAndSaveToDisk()
