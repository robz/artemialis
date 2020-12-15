import torch

from MaestroMidiDataset import get_dataloader, preprocessAndSaveToDisk
from PerformanceRNN import PerformanceRNN
from Train import train
from EpochWriter import EpochWriter
from MidiIndexUtils import NUM_CHANNELS


#preprocessAndSaveToDisk()
  
lstm = PerformanceRNN(channels=NUM_CHANNELS, hidden_size=512, num_layers=3).to('cuda')

prime = torch.zeros(1, 1, 388).to('cuda')
epochWriter = EpochWriter(
  model=lstm,
  name_prefix='performance_rnn',
  get_seq_for_errors=lambda lstm: lstm.forward_step(1000, prime=prime)
)
epochWriter.get_latest_model(iteration_override=0)

train(
  model=lstm,
  dataloaders={
    phase: get_dataloader(phase, len=12, max_data_len=2048) 
    for phase in ['train', 'eval']
  },
  num_epochs=500,
  lr=1e-3,
  epoch_loss_cb=lambda phase, loss: epochWriter.write_data_epoch(phase, loss)
)












