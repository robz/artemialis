import torch
import platform, time
import matplotlib.pyplot as plt

from MidiIndexUtils import writeMidi, idxsToMidi, NUM_CHANNELS, idx_to_event
from MaestroMidiDatasetWithConditioning import MaestroMidiDatasetWithConditioning, get_condition
from Wavenet import Wavenet

wavenet = Wavenet(
  input_channels=NUM_CHANNELS,
  output_channels=NUM_CHANNELS,
  hidden_channels=128,
  num_layers=10,
  num_stacks=2,
  kernel_size=2,
  dilation_rate=2,
).to('cuda')
wavenet.eval()

prime = torch.zeros(1, NUM_CHANNELS, wavenet.receptive_field, device='cuda')

print('predicting...')

start = time.time()
y = wavenet.fast_forward_steps(prime, 3)
y = y.to('cpu').long().detach().numpy()
print('delta', time.time() - start)


#start = time.time()
#y = wavenet.slow_forward_steps(prime.transpose(1, 2), 1000)
#y = y.to('cpu').long().detach().numpy()
#print('delta', time.time() - start)

print('done')
