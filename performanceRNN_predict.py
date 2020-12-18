import torch
import platform
import matplotlib.pyplot as plt

from PerformanceRNN import PerformanceRNN
from MidiIndexUtils import writeMidi, idxsToMidi, NUM_CHANNELS
from MaestroMidiDataset import MaestroMidiDataset

top_dir = '' if platform.system() == 'Linux' else 'Documents/PerformanceMidi'

model_path = top_dir + 'models/performance_rnn-iter4-221.pt'
midi_path = top_dir + 'music/performance_rnn-iter4-221.pt.midi'
#model_path = top_dir + 'final_models/performance_rnn-iter2-186.pt'
#midi_path = top_dir + 'music/performance_rnn-iter2-186.pt.midi'
#model_path = top_dir + 'models/performance_rnn-iter9-596.pt'
#midi_path = top_dir + 'music/performance_rnn-iter9-596.pt.midi'

print('creating and loading model...')
lstm = PerformanceRNN(channels=NUM_CHANNELS, hidden_size=1024, num_layers=3, dropout=0.5).to('cuda')
lstm.load_state_dict(torch.load(model_path))
lstm.eval()

print('predicting...')

#prime = torch.zeros(1, 1, 388)
#prime[0, 0, 355] = 1

dataset = MaestroMidiDataset('train')
dataset.fillCache()
prime = dataset.cache[dataset.df['midi_filename'][0]][:100].long()
onehot = torch.eye(NUM_CHANNELS)
prime = onehot[prime][None, :, :]

#print(dataset.df['midi_filename'][0])
#y = prime

y = lstm.forward_step(10000, prime=prime.to('cuda'), greedy=False)
y = y.to('cpu').long().detach().numpy()

print('writing to midi and plotting...')
mf, errors = idxsToMidi(y)
print(errors)
writeMidi(mf, filename=midi_path)

plt.plot(y)
plt.show()
