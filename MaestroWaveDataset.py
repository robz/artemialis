import random
import pandas as pd
import torch
import torchaudio
import os

from MaestroMidiDataset import MAESTRO_DIRECTORY


####################################################################
# Dataset class, dataloader, data augmentation
####################################################################


class MaestroWaveDataset(torch.utils.data.Dataset):
  # phase is train or eval
  def __init__(self, directory=MAESTRO_DIRECTORY):
    self.directory = directory
    self.df = pd.read_csv(directory + '/maestro-v2.0.0.csv')

  def preprocessAndSaveToDisk(self):
    length = len(self)
    output_directory = self.directory + '-tensors/'
    os.makedirs(output_directory, exist_ok=True)
    mulaw = torchaudio.transforms.MuLawEncoding()
    resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
    for i in range(length):
      filename = self.df['audio_filename'][i]
      x, rate = torchaudio.load(self.directory + '/' + filename)
      x = mulaw(resample(x[0])).byte()
      os.makedirs(output_directory + os.path.dirname(filename), exist_ok=True)
      torch.save(x, output_directory + filename + '.pt')
      if i % 10 == 0:
        print(i + 1, '/', length)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, i):
    return 'todo'


MaestroWaveDataset().preprocessAndSaveToDisk()
