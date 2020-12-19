import torch
import platform
import matplotlib.pyplot as plt

from MidiIndexUtils import writeMidi, idxsToMidi, NUM_CHANNELS, idx_to_event
from MaestroMidiDatasetWithConditioning import MaestroMidiDatasetWithConditioning, get_condition
from Wavenet import Wavenet

top_dir = '' if platform.system() == 'Linux' else 'Documents/PerformanceMidi/'

model_path = top_dir + 'models/performance_wavenet-iter15-499.pt'
midi_path = top_dir + 'music/performance_wavenet-iter15-499.pt.midi'

wavenet = Wavenet(
  input_channels=NUM_CHANNELS + 128,
  output_channels=NUM_CHANNELS,
  hidden_channels=128,
  num_layers=10,
  num_stacks=2,
  kernel_size=2,
  dilation_rate=2,
).to('cuda')
wavenet.load_state_dict(torch.load(model_path))
wavenet.eval()

dataset = MaestroMidiDatasetWithConditioning('train')
dataset.fillCache()
prime = dataset.getFullItem(0, wavenet.receptive_field)['input'].T[None, :, :].to('cuda')
prime_condition = prime[0, -128:, -1]

print('predicting...')

condition, inputs = get_condition(prime_condition, record_inputs=False)
#temp = torch.zeros(NUM_CHANNELS + 128, device='cuda')
#condition = lambda event: temp

y = wavenet.slow_forward_steps(prime.transpose(1, 2), 10000, greedy=False, condition=condition)

y = y.to('cpu').long().detach().numpy()

# print out inputs and conditioning 
#inputs = [e.to('cpu') for e in inputs]
#for i in range(len(y)):
#  notes_on = torch.nonzero(inputs[i])[:, 0]
#  print(i, idx_to_event[y[i]], list(notes_on.numpy()))

print('writing to midi and plotting...')
mf, errors = idxsToMidi(y)
print(errors)
writeMidi(mf, filename=midi_path)

plt.plot(y)
plt.show()
