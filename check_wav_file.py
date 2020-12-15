import torchaudio

data, rate = torchaudio.load('performance_rnn-iter2-zeros.mid.wav')

print(data.shape, rate)
