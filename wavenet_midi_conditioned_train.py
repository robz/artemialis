import torch

from MidiIndexUtils import NUM_CHANNELS
from MaestroMidiDatasetWithConditioning import MaestroMidiDatasetWithConditioning, get_condition
from Wavenet import Wavenet
from Train import train
from EpochWriter import EpochWriter

wavenet = Wavenet(
  input_channels=NUM_CHANNELS + 128,
  output_channels=NUM_CHANNELS,
  hidden_channels=128,
  num_layers=10,
  num_stacks=4,
  kernel_size=2,
  dilation_rate=2,
).to('cuda')
print('receptive field:', wavenet.receptive_field)

prime = torch.zeros(1, NUM_CHANNELS + 128, wavenet.receptive_field).to('cuda').transpose(1, 2)
epochWriter = EpochWriter(
  model=wavenet,
  name_prefix='performance_wavenet',
  get_seq_for_errors=lambda wavenet: wavenet.slow_forward_steps(prime, 1000, condition=get_condition()[0]),
  iteration=17
)
#epochWriter.get_latest_model()
#epochWriter.get_model(6, 12)
train_losses, test_losses = train(
  model=wavenet,
  dataloaders={
    phase: MaestroMidiDatasetWithConditioning.get_dataloader(phase, max_data_len=1024 * 4)
    for phase in ['train', 'eval']
  },
  num_epochs=500,
  lr=1e-3,
  epoch_loss_cb=lambda phase, loss, all_losses: epochWriter.write_data_epoch(phase, loss, all_losses)
)
