import torch
import torch.nn.functional as F

from MidiIndexUtils import NUM_CHANNELS


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
    y, _ = self.lstm(x)
    y = self.linear(F.relu(y))
    # transposing here is necessary because
    # 1. torch.nn.CrossEntropyLoss expects the output as [batch, channels, seq]
    # 2. torch.nn.Conv1d (used by Wavenet) expects [batch, channels, seq]
    return y.transpose(1, 2)

  # x is [batch, seq, channels] 
  def forward_hidden(self, x, hidden=None):
    y, hidden = self.lstm(x, hidden)
    y = self.linear(F.relu(y))
    return y, hidden

  # prime is [1, seq, channels]
  def forward_step(self, steps, prime=torch.zeros(1, 1, 388), greedy=True, alpha=1.0):
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
