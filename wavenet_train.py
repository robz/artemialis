import torch

from MidiIndexUtils import NUM_CHANNELS
from MaestroMidiDatasetWithConditioning import MaestroMidiDatasetWithConditioning
from Wavenet import Wavenet
from Train import train
from EpochWriter import EpochWriter


transition_matrix = torch.cat([
  torch.diag(torch.ones(128)), 
  torch.diag(torch.empty(128).fill_(-1)), 
  torch.zeros(128, 388-256)
], axis=1).to('cuda')


def get_condition(prime=None):
  # compute current conditioning from prime
  if prime is not None:
    current_notes_on = [prime[0, -1, -128:]]
  else:
    current_notes_on = [torch.zeros(128, device='cuda')]

  inputs = []

  def condition(x):
    event = x[0, :, 0]
    # update current conditioning based on latest event
    current_notes_on[0] = torch.clamp(current_notes_on[0] + torch.matmul(transition_matrix, event), 0, 1)
    # concat it to the input and return it
    input = torch.cat([event, current_notes_on[0]])[None, :, None]
    #inputs.append(input)
    return input

  return condition, inputs


wavenet = Wavenet(
  input_channels=NUM_CHANNELS + 128,
  output_channels=NUM_CHANNELS,
  hidden_channels=64,
  num_layers=9,
  num_stacks=4,
  kernel_size=2,
  dilation_rate=2,
).to('cuda')
print('receptive field:', wavenet.receptive_field)


prime = torch.zeros(1, NUM_CHANNELS + 128, wavenet.receptive_field).to('cuda')
epochWriter = EpochWriter(
  model=wavenet,
  name_prefix='performance_wavenet',
  get_seq_for_errors=lambda wavenet: wavenet.fast_forward_steps(prime, 1000, condition=get_condition()[0]),
  iteration=14
)
#epochWriter.get_latest_model()
#epochWriter.get_model(6, 12)
train_losses, test_losses = train(
  model=wavenet,
  dataloaders={
    phase: MaestroMidiDatasetWithConditioning.get_dataloader(phase, max_data_len=2048)
    for phase in ['train', 'eval']
  },
  num_epochs=500,
  lr=1e-3,
  epoch_loss_cb=lambda phase, loss, all_losses: epochWriter.write_data_epoch(phase, loss, all_losses)
)
