import torch
import torch.nn.functional as F
import torchaudio
from GPUUsageUtils import printm

class CausalConvLayer(torch.nn.Module):
  def __init__(self, channels, dilation, kernel_size):
    super(CausalConvLayer, self).__init__()
    self.channels = channels
    self.dilation = dilation
    self.padding = (kernel_size - 1) * dilation
    self.filter_gate_conv = torch.nn.Conv1d(
      in_channels=channels, 
      out_channels=channels * 2, 
      kernel_size=kernel_size, 
      dilation=dilation,
    )
    self.residual_conv = torch.nn.Conv1d(
      in_channels=channels,
      out_channels=channels,
      kernel_size=1,
    )
    self.skip_conv = torch.nn.Conv1d(
      in_channels=channels,
      out_channels=channels,
      kernel_size=1,
    )

  def forward_unpadded(self, x):
    filter_gate = self.filter_gate_conv(x)
    filter = filter_gate[:, :self.channels, :]
    gate = filter_gate[:, self.channels:, :]
    z = torch.tanh(filter) * torch.sigmoid(gate)
    residual = x[:, :, -z.shape[2]:] + self.residual_conv(z)
    skip = self.skip_conv(z)
    return residual, skip

  # using left padding causes the output to have the same length as the input
  def forward(self, x, use_padding):
    x = F.pad(x, (self.padding, 0)) if use_padding else x
    return self.forward_unpadded(x)

class Wavenet(torch.nn.Module):
  def __init__(
    self, 
    input_channels,
    output_channels,
    hidden_channels,
    num_layers,
    num_stacks,
    kernel_size,
    dilation_rate,
    device="cuda",
  ):
    super(Wavenet, self).__init__()

    self.input_to_hidden_conv = torch.nn.Conv1d(
      in_channels=input_channels,
      out_channels=hidden_channels,
      kernel_size=1
    )

    layers = []
    for _ in range(num_stacks):
      for layer in range(num_layers):
        layers.append(CausalConvLayer(
          channels=hidden_channels, 
          dilation=dilation_rate ** layer,
          kernel_size=kernel_size,
        ))
    self.layers = torch.nn.ModuleList(layers)

    self.conv1 = torch.nn.Conv1d(
      in_channels=hidden_channels,
      out_channels=hidden_channels,
      kernel_size=1
    )

    self.conv2 = torch.nn.Conv1d(
      in_channels=hidden_channels,
      out_channels=output_channels,
      kernel_size=1
    )

    self.kernel_size = kernel_size
    self.receptive_field = num_stacks * (dilation_rate ** num_layers - 1) * (kernel_size - 1) // (dilation_rate - 1) + 1

    self.device = device
    self.hidden_channels = hidden_channels
    self.onehot_input = torch.eye(input_channels, device=device)
    self.onehot_output = torch.eye(output_channels, device=device)
    self.mulawEncode = torchaudio.transforms.MuLawEncoding()
    self.mulawDecode = torchaudio.transforms.MuLawDecoding()

  # x is [batch, seq, channels]
  # returns [batch, channels, seq]
  # using left padding causes the output to have the same length as the input
  def forward(self, x, use_padding=True):
    x = x.transpose(1, 2) # transpose to [batch, channels, seq]
    x = self.input_to_hidden_conv(x)
    skip_sum = torch.zeros_like(x)
    for layer in self.layers:
      x, skip = layer(x, use_padding=use_padding)
      skip_sum = skip_sum[:, :, -skip.shape[2]:] + skip
    y = self.conv1(F.relu(skip_sum))
    y = self.conv2(F.relu(y))
    return y

  # x is [batch, seq, channels]
  def slow_forward_steps(self, x, steps, greedy=True, condition=None):
    outputs = torch.zeros((steps), device=self.device)
    printm()

    for i in range(steps):
      ypred = self.forward(x, use_padding=False)
      ypred = ypred[0, :, -1]
      sample = torch.argmax(ypred) if greedy else torch.distributions.Categorical(torch.softmax(ypred, 0)).sample()
      outputs[i] = sample
      sample = self.onehot_output[sample]
      if condition is not None:
        sample = condition(sample)
      sample = sample[None, None, :]      
      x = torch.cat([x[:, 1:, :], sample], axis=1)
    printm()

    return outputs

  # x is [batch, channels, seq]
  # https://arxiv.org/abs/1611.09482
  def fast_forward_steps(self, x, steps, greedy=True, alpha=1.0, condition=None):
    queues = [torch.zeros((1, self.hidden_channels, 0), device=self.device) for _ in self.layers]
    outputs = torch.zeros((steps), device=self.device)

    prev = [torch.cuda.memory_allocated('cuda')]
    def print_mem(s):
        alloc = torch.cuda.memory_allocated('cuda')
        print(s, alloc, alloc - prev[0])
        prev[0] = alloc

    # the first step uses the full receptive field
    # subsequent steps use only the end of the previous output
    for j in range(steps):
      x = self.input_to_hidden_conv(x)
      skip_sum = 0.0
      for i, layer in enumerate(self.layers):
        # TODO: there is a memory leak here
        x = torch.cat([queues[i], x], axis=2)
        queues[i] = x[:, :, -layer.dilation * (self.kernel_size - 1):]
        x, skip = layer.forward_unpadded(x)
        skip_sum += skip[:, :, -1:]
      y = self.conv1(F.relu(skip_sum))
      y = self.conv2(F.relu(y))
      y = y[0, :, -1]
      sample = torch.argmax(y) if greedy else torch.distributions.Categorical(torch.softmax(alpha * y, 0)).sample()
      outputs[j] = sample
      x = self.onehot_output[sample]
      if condition is not None:
        x = condition(x)
      x = x[None, :, None]

    printm()

    return outputs

  # x is [seq]
  def forward_encode_nomulaw(self, x):
    x = self.onehot_input[x][None, :, :]
    output = self.forward(x, use_padding=True)
    output = torch.argmax(output.squeeze(), 0).to('cpu')
    return output

  def forward_encode(self, x):
    x = self.onehot_input[self.mulawEncode(x)].T[None, :, :]
    output = self.forward(x, use_padding=True)
    output = self.mulawDecode(torch.argmax(output.squeeze(), 0)).to('cpu')
    return output

  def fast_forward_steps_encode_nomulaw(self, x, steps, greedy=True, alpha=1.0, condition=None):
    x = self.onehot_input[x].T[None, :, :]
    output = self.fast_forward_steps(x, steps, greedy, alpha, condition)
    output = output.to('cpu')
    return output

  def fast_forward_steps_encode(self, x, steps, greedy=True):
    x = self.onehot_input[self.mulawEncode(x)].T[None, :, :]
    output = self.fast_forward_steps(x, steps, greedy)
    output = self.mulawDecode(output).to('cpu')
    return output
