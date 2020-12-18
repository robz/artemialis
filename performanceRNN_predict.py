import torch
import matplotlib.pyplot as plt

from PerformanceRNN import PerformanceRNN
from MidiIndexUtils import writeMidi, idxsToMidi, NUM_CHANNELS
from MaestroMidiDataset import MaestroDataset

model_path = 'Documents/PerformanceMidi/models/performance_rnn-iter4-129.pt'
midi_path = 'Documents/PerformanceMidi/music/performance_rnn-iter4-129.pt.midi'
#model_path = 'Documents/PerformanceMidi/final_models/performance_rnn-iter2-186.pt'
#midi_path = 'Documents/PerformanceMidi/music/performance_rnn-iter2-186.pt.midi'
#model_path = 'Documents/PerformanceMidi/models/performance_rnn-iter9-596.pt'
#midi_path = 'Documents/PerformanceMidi/music/performance_rnn-iter9-596.pt.midi'

print('creating and loading model...')
lstm = PerformanceRNN(channels=NUM_CHANNELS, hidden_size=1024, num_layers=3, dropout=0.5).to('cuda')
lstm.load_state_dict(torch.load(model_path))
lstm.eval()

print('predicting...')

#prime = torch.zeros(1, 1, 388)
#prime[0, 0, 355] = 1

dataset = MaestroDataset('train')
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
