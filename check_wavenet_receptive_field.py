import torch

from Wavenet import Wavenet
from MidiIndexUtils import NUM_CHANNELS

wavenet = Wavenet(input_channels=NUM_CHANNELS, output_channels=NUM_CHANNELS, hidden_channels=64, num_layers=10, num_stacks=2, kernel_size=2, dilation_rate=2, device='cpu')
wavenet.load_state_dict(torch.load("Documents/PerformanceMidi/models/performance_wavenet-iter10-400.pt"))
print(wavenet.receptive_field)

wavenet = Wavenet(input_channels=NUM_CHANNELS, output_channels=NUM_CHANNELS, hidden_channels=64, num_layers=9, num_stacks=4, kernel_size=2, dilation_rate=2, device='cpu')
wavenet.load_state_dict(torch.load("Documents/PerformanceMidi/models/performance_wavenet-iter11-400.pt"))
print(wavenet.receptive_field)

wavenet = Wavenet(input_channels=NUM_CHANNELS+128, output_channels=NUM_CHANNELS, hidden_channels=64, num_layers=10, num_stacks=2, kernel_size=2, dilation_rate=2, device='cpu')
wavenet.load_state_dict(torch.load("Documents/PerformanceMidi/models/performance_wavenet-iter13-400.pt"))
print(wavenet.receptive_field)

wavenet = Wavenet(input_channels=NUM_CHANNELS+128, output_channels=NUM_CHANNELS, hidden_channels=64, num_layers=9, num_stacks=4, kernel_size=2, dilation_rate=2, device='cpu')
wavenet.load_state_dict(torch.load("Documents/PerformanceMidi/models/performance_wavenet-iter14-1.pt"))
print(wavenet.receptive_field)

wavenet = Wavenet(input_channels=NUM_CHANNELS+128, output_channels=NUM_CHANNELS, hidden_channels=128, num_layers=10, num_stacks=2, kernel_size=2, dilation_rate=2, device='cpu')
wavenet.load_state_dict(torch.load("Documents/PerformanceMidi/models/performance_wavenet-iter15-1.pt"))
print(wavenet.receptive_field)

wavenet = Wavenet(input_channels=NUM_CHANNELS+128, output_channels=NUM_CHANNELS, hidden_channels=128, num_layers=10, num_stacks=4, kernel_size=2, dilation_rate=2, device='cpu')
wavenet.load_state_dict(torch.load("Documents/PerformanceMidi/models/performance_wavenet-iter17-1.pt"))
print(wavenet.receptive_field)
