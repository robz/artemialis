import torch
from torch.utils.tensorboard import SummaryWriter
import os, re, platform, math

from MidiIndexUtils import writeMidi, idxsToMidi
from GPUUsageUtils import printm

DEFAULT_MODEL_DIR = 'models' if platform.system() == 'Linux' else 'Documents/PerformanceMidi/models'

DEFAULT_RUNS_DIR = '../runs' if platform.system() == 'Linux' else 'runs'

class EpochWriter:
  def __init__(self,
    model,
    name_prefix,
    get_seq_for_errors,
    iteration=0,
    model_dir=DEFAULT_MODEL_DIR
  ):
    self.model = model
    self.name_prefix = name_prefix
    self.get_seq_for_errors = get_seq_for_errors
    self.model_dir = model_dir
    
    self.epoch = 0
    self.name = F'{self.name_prefix}-iter{iteration}'
    self.writer = SummaryWriter(DEFAULT_RUNS_DIR + '/' + self.name)

  def to_tup(self, filename):
    search = re.search(F'{self.name_prefix}-iter([0-9]+)-([0-9]+).pt', filename)
    return search.group(1), search.group(2)

  def get_latest_model(self, iteration_override=None):
    files = filter(lambda f: self.name_prefix in f, os.listdir(self.model_dir))
    (iteration, epoch) = sorted(
      [self.to_tup(name) for name in files],
      key=lambda x: int(x[0]) * 10000 + int(x[1]),
      reverse=True
    )[0]

    name = F'{self.name_prefix}-iter{iteration}'
    self.model.load_state_dict(torch.load(F"{self.model_dir}/{name}-{epoch}.pt"))

    self.epoch = 1 + int(epoch)
    iteration = iteration if iteration_override is None else iteration_override
    self.name = F'{self.name_prefix}-iter{iteration}'
    self.writer = SummaryWriter(DEFAULT_RUNS_DIR + '/' + self.name)

  def get_model(self, iteration, epoch):
    self.model.load_state_dict(torch.load(
      F"{self.model_dir}/{self.name_prefix}-iter{iteration}-{epoch}.pt"
    ))

  def write_data_epoch(self, phase, loss_avg, all_losses):
    self.writer.add_scalar(phase + '_epoch_loss', loss_avg, self.epoch)
    print(self.epoch, phase, loss_avg) 

    start = self.epoch * len(all_losses)
    for i, loss in enumerate(all_losses):
      self.writer.add_scalar(phase + '_loss', loss, i + start)

    if math.isnan(loss_avg):
      print('loss is nan, quitting!')
      exit()

    if phase == 'train':
      self.model.eval()
      print('getting prediction for errors...')
      y = self.get_seq_for_errors(self.model)
      y = y.to('cpu').long().detach().numpy()
      mf, errors = idxsToMidi(y)
      num_errors = sum(errors.values())
      self.writer.add_scalar('train_epoch_errors', num_errors, self.epoch)
      
      mem_util_pct = printm()
      self.writer.add_scalar('train_epoch_gpu_usage', mem_util_pct, self.epoch)
      
      #writeMidi(mf, filename=F"Documents/PerformanceMidi/music/{self.name}-{self.epoch}.mid")
      torch.save(self.model.state_dict(), F"{self.model_dir}/{self.name}-{self.epoch}.pt")
    else:
      self.epoch += 1
