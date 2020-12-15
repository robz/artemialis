import music21
from music21 import midi
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import math
import random
import time

df = pd.read_csv('Documents/PerformanceMidi/maestro-v2.0.0.csv')



####################################################################
# Functions to translate from midi files to sequences and back
####################################################################

def readMidi(filepath):
  mf = midi.MidiFile()
  mf.open(filepath)
  mf.read()
  mf.close()
  return mf


def writeMidi(mf, filename = 'tempmidi.mid'):
  mf.open(filename, attrib='wb')
  mf.write()
  mf.close()
  return filename

"""
def playMidi(mf):
  filename = writeMidi(mf)
  !fluidsynth -ni font.sf2 $filename -F $filename\.wav -r 16000 > /dev/null
  display(Audio(filename + '.wav'))
"""

event_to_idx = {}
for i in range(128):
  event_to_idx['note-on-' + str(i)] = i
for i in range(128):
  event_to_idx['note-off-' + str(i)] = i + 128
for i in range(100):
  event_to_idx['time-shift-' + str(i + 1)] = i + 128 + 128
for i in range(32):
  event_to_idx['velocity-' + str(i)] = i + 128 + 128 + 100


idx_to_event = list(event_to_idx.keys())
NUM_CHANNELS = len(idx_to_event)

def midiToIdxs(mf):
  ticks_per_beat = mf.ticksPerQuarterNote
  # The maestro dataset uses the first track to store tempo data
  tempo_data = next(e for e in mf.tracks[0].events if e.type == 'SET_TEMPO').data
  # tempo data is stored at microseconds per beat (beat = quarter note)
  microsecs_per_beat = int.from_bytes(tempo_data, 'big')
  millis_per_tick = microsecs_per_beat / ticks_per_beat / 1e3

  idxs = []
  started = False
  previous_t = None

  # The second track stores the actual performance
  for e in mf.tracks[1].events:
    if started and e.type == 'DeltaTime' and e.time > 0:
      # event times are stored as ticks, so convert to milliseconds
      millis = e.time * millis_per_tick

      # combine repeated delta time events
      t = millis + (0 if previous_t is None else previous_t)

      # we can only represent a max time of 1 second (1000 ms)
      # so we must split up times that are larger than that into separate events
      while t > 0:
        t_chunk = min(t, 1000)
        idx = event_to_idx['time-shift-' + str(math.ceil(t_chunk / 10))]
        if previous_t is None:
          idxs.append(idx)
        else:
          idxs[-1] = idx
          previous_t = None
        t -= t_chunk
      previous_t = t_chunk

    elif e.type == 'NOTE_ON':
      if e.velocity == 0:
        idxs.append(event_to_idx['note-off-' + str(e.pitch)])
      else:
        # midi supports 128 velocities, but our representation only allows 32
        idxs.append(event_to_idx['velocity-' + str(e.velocity // 4)])
        idxs.append(event_to_idx['note-on-' + str(e.pitch)])
        started = True
      previous_t = None

  return idxs


def idxsToMidi(idxs, verbose=False):
  mf = midi.MidiFile()
  mf.ticksPerQuarterNote = 1024
  
  # The maestro dataset uses the first track to store tempo data, and the second
  # track to store the actual performance. So follow that convention.
  tempo_track = midi.MidiTrack(0)
  track = midi.MidiTrack(1)
  mf.tracks = [tempo_track, track]

  tempo = midi.MidiEvent(tempo_track, type='SET_TEMPO')
  # temp.data is the number of microseconds per beat (per quarter note)
  # So to set ticks per millis = 1 (easy translation from time-shift values to ticks),
  # tempo.data must be 1e3 * 1024, since ticksPerQuarterNote is 1024 (see above)
  tempo.data = int(1e3 * 1024).to_bytes(3, 'big')

  end_of_track = midi.MidiEvent(tempo_track, type='END_OF_TRACK')
  end_of_track.data = ''
  tempo_track.events = [
    # there must always be a delta time before each event
    midi.DeltaTime(tempo_track, time=0),
    tempo,
    midi.DeltaTime(tempo_track, time=0),
    end_of_track
  ]

  track.events = [midi.DeltaTime(track, time=0)]

  current_velocity = 0
  notes_on = set()

  for idx in idxs:
    if 0 <= idx < 128: # note-on
      pitch = idx
      if pitch in notes_on:
        if verbose:
          print(pitch, 'is already on')
        continue
      if track.events[-1].type != 'DeltaTime':
        track.events.append(midi.DeltaTime(track, time=0))
      e = midi.MidiEvent(track, type='NOTE_ON', channel=1)
      e.pitch = pitch
      e.velocity = current_velocity
      track.events.append(e)
      notes_on.add(pitch)

    elif 128 <= idx < (128 + 128): # note-off
      pitch = idx - 128
      if pitch not in notes_on:
        if verbose:
          print(pitch, 'is not on')
        continue
      if track.events[-1].type != 'DeltaTime':
        track.events.append(midi.DeltaTime(track, time=0))
      e = midi.MidiEvent(track, type='NOTE_ON', channel=1)
      e.pitch = pitch
      e.velocity = 0
      track.events.append(e)
      notes_on.remove(pitch)

    elif (128 + 128) <= idx < (128 + 128 + 100): # time-shift
      t = (1 + idx - (128 + 128)) * 10
      if track.events[-1].type == 'DeltaTime':
        # combine repeated delta times
        track.events[-1].time += t
      else:
        track.events.append(midi.DeltaTime(track, time=t))

    else: # velocity
      current_velocity = (idx - (128 + 128 + 100)) * 4

  if verbose:
    print('remaining notes left on:', notes_on)

  if track.events[-1].type != 'DeltaTime':
    track.events.append(midi.DeltaTime(track, time=0))

  e = midi.MidiEvent(track, type='END_OF_TRACK')
  e.data = ''
  track.events.append(e)

  return mf


"""
first_filename = 'Music/maestro-v2.0.0/' +  df['midi_filename'][0]
midi_data = readMidi(first_filename)
midi_tensor = torch.tensor(midiToIdxs(midi_data))
print("midi_tensor.shape", midi_tensor.shape)
"""
####################################################################
# Dataset class and dataloader
####################################################################

class MaestroDataset(torch.utils.data.Dataset):
  # phase is train or eval
  def __init__(self, phase, channels=NUM_CHANNELS, len=None):
    df = pd.read_csv('Documents/PerformanceMidi/maestro-v2.0.0.csv')
    self.df = df[(df['split'] == 'train') if phase == 'train' else (df['split'] != 'train')]
    self.df.reset_index(drop=True, inplace=True)
    self.onehot = torch.eye(channels)
    self.manual_len = len

  def __len__(self):
    return self.manual_len if self.manual_len is not None else len(self.df)

  def __getitem__(self, idx):
    filename = 'Music/maestro-v2.0.0/' +  self.df['midi_filename'][idx]
    midi_data = readMidi(filename)
    x = torch.tensor(midiToIdxs(midi_data)[:1024])
    input = self.onehot[x[:-1]]
    output = x[1:]
    return {'input': input, 'output': output}


def my_collate(data):
  min_len = min([e['output'].shape[0] for e in data])
  return {
    'output': torch.stack([e['output'][:min_len] for e in data]), 
    'input': torch.stack([e['input'][:min_len, :] for e in data]),
  }


def get_dataloader(phase, len=None):
  return torch.utils.data.DataLoader(
    MaestroDataset(phase, len=len), 
    batch_size=4,
    shuffle=True, 
    num_workers=0, 
    collate_fn=my_collate
  )


"""
dataset = MaestroDataset('train')
for i in range(5):
  plt.plot(dataset[i]['output'][:30])
plt.show()

dataloader = get_dataloader('train', len=12)
start = time.time()
for i_batch, sample_batched in enumerate(dataloader):
  print(i_batch, sample_batched['input'].shape)
print(time.time() - start)
"""





####################################################################
# PerformanceRNN
####################################################################

class PerformanceRNN(torch.nn.Module):
  def __init__(self, channels, hidden_size, num_layers):
    super(PerformanceRNN, self).__init__()
    self.lstm = torch.nn.LSTM(input_size=channels, hidden_size=hidden_size, num_layers=3, batch_first=True)
    self.linear = torch.nn.Linear(in_features=hidden_size, out_features=channels)
    self.hidden_size = hidden_size

  # x is [batch, seq, channels]
  def forward(self, x):
    y = self.lstm(x)[0]
    y = self.linear(F.relu(y))
    return y

  # x is [batch, seq, channels] 
  def forward_hidden(self, x, hidden=None):
    y, hidden = self.lstm(x, hidden)
    y = self.linear(F.relu(y))
    return y, hidden

  # prime is [1, seq, channels]
  def forward_step(self, steps, prime=torch.zeros(1, 1, 388), greedy=False, alpha=1.0):
    device = next(self.parameters()).device
    onehot = torch.eye(NUM_CHANNELS).to(device)
    result = torch.zeros(steps).to(device)

    output, hidden = self.forward_hidden(prime)

    element = torch.argmax(output[0, -1], 0) if greedy else torch.distributions.Categorical(torch.softmax(alpha * output[0, -1], 0)).sample()
    result[0] = element

    for i in range(steps - 1):
      input = onehot[element][None, None, :]
      output, hidden = self.forward_hidden(input, hidden)
      element = torch.argmax(output[0, -1], 0) if greedy else torch.distributions.Categorical(torch.softmax(alpha * output[0, -1], 0)).sample()
      result[i + 1] = element

    return result.to('cpu')

"""
x = torch.tensor(midiToIdxs(readMidi(first_filename)))[:1000]
onehot = torch.eye(NUM_CHANNELS)
x = onehot[x][None, :, :]
print('input to lstm', x.shape)

lstm = PerformanceRNN(channels=NUM_CHANNELS, hidden_size=512, num_layers=3)
y = lstm(x)
print('output of lstm', y.shape)
"""




####################################################################
# Training loop
####################################################################

def train(model, dataloaders, num_epochs, lr=1e-3, test_cb=lambda model, i:None):
  optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
  lossfn = torch.nn.CrossEntropyLoss()

  train_losses = []
  test_losses = []

  for epoch in range(num_epochs):
    start = time.time()
    for phase in ['train', 'eval']:
      epoch_losses = []
      if phase == 'train':
        model.train()
      else:
        model.eval()

      for sample_batched in dataloaders[phase]:
        x = sample_batched['input'].to('cuda')
        y = sample_batched['output'].to('cuda')
        ypred = model(x)
        loss = lossfn(ypred.transpose(1, 2), y[:, -ypred.shape[1]:])
        loss_item = loss.item()
        
        if phase == 'train':
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          epoch_losses.append(loss_item)
        else:
          test_cb(model, i)
          epoch_losses.append(loss_item)

        print(loss_item)
        
      loss_avg = sum(epoch_losses) / len(epoch_losses)
      (train_losses if phase == 'train' else test_losses).append(loss_avg)
      print(epoch, phase, loss_avg)
    print("epoch time elapsed:", time.time() - start)

  return train_losses, test_losses


def train_and_show(model, num_epochs=10):
  dataloaders = {phase: get_dataloader(phase) for phase in ['train', 'eval']}

  start = time.time()
  train_losses, test_losses = train(model, dataloaders, num_epochs=num_epochs, lr=1e-3)
  print("time elapsed:", time.time() - start)

  plt.title('loss over epoch')
  plt.plot(train_losses, label='train')
  plt.plot(test_losses, label='test')
  plt.legend()
  title = 'Documents/PerformanceMidi/performance-rnn-%s.png' % time.strftime('%m-%d-%Y-%H-%M-%S')
  plt.savefig(title)
  plt.close()


model_name = 'performance_rnn-iter8-507'
path = F"Documents/PerformanceMidi/models/{model_name}.pt"

lstm = PerformanceRNN(channels=NUM_CHANNELS, hidden_size=512, num_layers=3).to('cuda')
lstm.load_state_dict(torch.load(path))

lstm.eval()



## Single sequence from training
onehot = torch.eye(NUM_CHANNELS).to('cuda')
first_filename = 'Music/maestro-v2.0.0/' +  df['midi_filename'][400]
print(first_filename)
xraw = torch.tensor(midiToIdxs(readMidi(first_filename)))[:1000].long()
x = onehot[xraw][None, :, :]
#yraw = lstm(x)
#y = torch.argmax(yraw.squeeze(), 1).to('cpu')

"""
plt.title('prediction from training set sequence')
plt.plot(xraw[1:][:100], label='input')
plt.plot(y[:-1][:100], label='output')
plt.legend()
plt.show()

plt.imshow(yraw[0, :300, :].T.to('cpu').detach().numpy())
plt.show()
"""


## Novel sequence

#prime = torch.zeros(1, 1, 388)

#prime = torch.tensor([[event_to_idx[e] for e in ['velocity-15', 'note-on-64', 'time-shift-50']]])
#prime = onehot[prime]

prime = x[:, :100, :]

y = lstm.forward_step(10000, prime=prime.to('cuda'))
y = y.to('cpu').long().detach().numpy()
plt.plot(y)
plt.show()
mf = idxsToMidi(y)
writeMidi(mf, filename="Documents/PerformanceMidi/"+model_name+"-3.mid")



