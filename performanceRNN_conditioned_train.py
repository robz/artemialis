import torch

from MaestroMidiDatasetWithConditioning import MaestroMidiDatasetWithConditioning, get_condition
from PerformanceRNN import PerformanceRNN
from Train import train
from EpochWriter import EpochWriter
from MidiIndexUtils import NUM_CHANNELS


lstm = PerformanceRNN(
  input_channels=NUM_CHANNELS + 128,
  output_channels=NUM_CHANNELS,
  hidden_size=1024,
  num_layers=3,
  dropout=0.5,
).to('cuda')

prime = torch.zeros(1, 1, NUM_CHANNELS + 128).to('cuda')
epochWriter = EpochWriter(
  model=lstm,
  name_prefix='performance_rnn_conditioned',
  get_seq_for_errors=lambda lstm: lstm.forward_step(1000, prime=prime, condition=get_condition()),
  iteration=1,
)
#epochWriter.get_latest_model(iteration_override=0)

train(
  model=lstm,
  dataloaders={
    phase: MaestroMidiDatasetWithConditioning.get_dataloader(phase, max_data_len=2048)
    for phase in ['train', 'eval']
  },
  num_epochs=1000,
  lr=1e-3,
  epoch_loss_cb=lambda phase, loss, all_losses: epochWriter.write_data_epoch(phase, loss, all_losses)
)
