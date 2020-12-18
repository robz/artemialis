import torch
import time, os, re
from torch.utils.tensorboard import SummaryWriter

from MidiIndexUtils import NUM_CHANNELS
from MaestroMidiDataset import get_dataloader, preprocessAndSaveToDisk
from Wavenet import Wavenet
from Train import train
from EpochWriter import EpochWriter


wavenet = Wavenet(
  input_channels=NUM_CHANNELS,
  hidden_channels=64,
  num_layers=9,
  num_stacks=4,
  kernel_size=2,
  dilation_rate=2,
).to('cuda')
print('receptive field:', wavenet.receptive_field)

prime = torch.zeros(1, NUM_CHANNELS, wavenet.receptive_field).to('cuda')
epochWriter = EpochWriter(
  model=wavenet,
  name_prefix='performance_wavenet',
  get_seq_for_errors=lambda wavenet: wavenet.fast_forward_steps(prime, 1000),
  iteration=11
)
#epochWriter.get_latest_model()
#epochWriter.get_model(6, 12)

train_losses, test_losses = train(
  model=wavenet,
  dataloaders={
    phase: get_dataloader(phase, max_data_len=2049) 
    for phase in ['train', 'eval']
  },
  num_epochs=500,
  lr=1e-3,
  epoch_loss_cb=lambda phase, loss, all_losses: epochWriter.write_data_epoch(phase, loss, all_losses)
)
