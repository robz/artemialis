import torch
import time, os, re
from torch.utils.tensorboard import SummaryWriter

from MidiIndexUtils import NUM_CHANNELS
from MaestroMidiDataset import get_dataloader, preprocessAndSaveToDisk
from Wavenet import Wavenet
from Train import train
from EpochWriter import EpochWriter


wavenet = Wavenet(channels=NUM_CHANNELS, num_layers=11, num_stacks=1, kernel_size=2, dilation_rate=2).to('cuda')


prime = torch.zeros(1, NUM_CHANNELS, wavenet.receptive_field).to('cuda')
epochWriter = EpochWriter(
  model=wavenet,
  name_prefix='performance_wavenet',
  get_seq_for_errors=lambda wavenet: wavenet.fast_forward_steps(prime, 1000),
  iteration=5
)
#epochWriter.get_latest_model()
#epochWriter.get_model(1, 150)

train_losses, test_losses = train(
  model=wavenet,
  dataloaders={
    phase: get_dataloader(phase, max_data_len=2049) 
    for phase in ['train', 'eval']
  },
  num_epochs=500,
  lr=1e-4,
  epoch_loss_cb=lambda phase, loss, all_losses: epochWriter.write_data_epoch(phase, loss, all_losses)
)
