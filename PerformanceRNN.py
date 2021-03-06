import torch
import torch.nn.functional as F

from MidiIndexUtils import NUM_CHANNELS


####################################################################
# PerformanceRNN
####################################################################

class PerformanceRNN(torch.nn.Module):
  def __init__(self, input_channels, output_channels, hidden_size, num_layers, dropout=0):
    super(PerformanceRNN, self).__init__()
    self.lstm = torch.nn.LSTM(
      input_size=input_channels,
      hidden_size=hidden_size,
      num_layers=3,
      batch_first=True,
      dropout=dropout
    )
    for name, param in self.lstm.named_parameters():
      if 'bias' in name:
         torch.nn.init.constant_(param, 0.0)
      elif 'weight' in name:
         torch.nn.init.xavier_normal_(param)
    self.linear = torch.nn.Linear(in_features=hidden_size, out_features=output_channels)
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
  def forward_step(self, steps, prime=torch.zeros(1, 1, 388), greedy=True, alpha=1.0, condition=None):
    device = next(self.parameters()).device
    onehot = torch.eye(NUM_CHANNELS).to(device)
    result = torch.zeros(steps).to(device)

    output, hidden = self.forward_hidden(prime)

    element = torch.argmax(output[0, -1], 0) if greedy else torch.distributions.Categorical(torch.softmax(alpha * output[0, -1], 0)).sample()
    result[0] = element

    for i in range(steps - 1):
      input = onehot[element]
      if condition is not None:
        input = condition(input)
      input = input[None, None, :]
      output, hidden = self.forward_hidden(input, hidden)
      element = torch.argmax(output[0, -1], 0) if greedy else torch.distributions.Categorical(torch.softmax(alpha * output[0, -1], 0)).sample()
      result[i + 1] = element

    return result.to('cpu')
