from Wavenet import Wavenet
from MidiIndexUtils import NUM_CHANNELS

wavenet = Wavenet(channels=NUM_CHANNELS, num_layers=10, num_stacks=2, kernel_size=2, dilation_rate=2, device='cpu')
print(wavenet.receptive_field)
