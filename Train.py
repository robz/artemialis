
import torch
import time, math
import sys


####################################################################
# Training loop
####################################################################

class BackwardError(Exception):
  pass

def train(
  model,
  dataloaders,
  num_epochs,
  lr=1e-3,
  loss_cb=None,
  epoch_loss_cb=None
):
  print('starting training!')
  optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
  lossfn = torch.nn.CrossEntropyLoss()

  train_losses = []
  test_losses = []

  epoch = 0
  while epoch < num_epochs:
    start = time.time()
    try:
      for phase in ['train', 'eval']:
        if phase == 'train':
          model.train()
        else:
          model.eval()
        print(F'{phase}: ', end='')  

        dataloader = dataloaders[phase]
        num_batches = int(math.ceil(len(dataloader.dataset) / dataloader.batch_size))
        epoch_losses = torch.zeros(num_batches).to('cuda')
        for i, sample_batched in enumerate(dataloader):
          x = sample_batched['input']
          y = sample_batched['output']
          print('.', end='')
          ypred = model(x)
          print('.', end='')
          loss = lossfn(ypred, y[:, -ypred.shape[2]:])
          
          print('.', end='')
          if phase == 'train':
            optimizer.zero_grad()
            
            try:
              loss.backward()
            except RuntimeError as err:
              # Vague cuda errors happen a lot :/
              print('\nCaught runtime error:')
              print(err)
              raise BackwardError()
            
            optimizer.step()

          epoch_losses[i] = loss.detach()

          if loss_cb is not None:
            loss_cb(phase, loss)

          print(F'{i}', end='')
          sys.stdout.flush()

        print()
          
        loss_avg = torch.sum(epoch_losses).item() / len(epoch_losses)
        (train_losses if phase == 'train' else test_losses).append(loss_avg)
        if epoch_loss_cb is not None:
          epoch_loss_cb(phase, loss_avg, epoch_losses.to('cpu'))
          
    except BackwardError:
      print("restarting epoch")
      time.sleep(30)
      continue
        
    print("epoch time elapsed:", time.time() - start)
    epoch += 1

  return train_losses, test_losses
