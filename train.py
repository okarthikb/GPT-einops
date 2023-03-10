import torch, os, pickle, random, wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import youtokentome as yttm
from gpt import *
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast


def process(rank, world_size):
  # initialize distributed process
  dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
  print(f'started process {rank}\n')

  # load dataset
  enc = pickle.load(open('c4-5e7-tokens.pkl', 'rb'))
  split_size = len(enc) // world_size
  enc = enc[:split_size * world_size]

  # dataset split for process
  split = torch.tensor(enc[rank * split_size:rank * split_size + split_size])
  
  # hyperparams
  d = 512
  nh = 16
  nl = 8
  l = 512
  v = yttm.BPE('c4-5e7-tokenizer.model').vocab_size()
  batch_size = 16
  steps = 1200
  lr = 1e-4

  # initialize model and optimizer
  torch.manual_seed(69)  # set seed so every process initializes same model
  gpt = GPT(d, nh, nl, l, v).cuda(rank)
  optimizer = Adam(gpt.parameters(), lr)

  # generator for data loading
  def generator():
    for _ in range(steps):
      indices = random.choices(range(len(split) - l - 1), k=batch_size)
      yield torch.stack([split[i:i + l + 1] for i in indices])

  # process 0 does logging
  if rank == 0:
    nparam = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
    print(f'{nparam} parameters\n')
    wandb.init(project='...')
    wandb.run.name = '...'
    print()

  # train loop
  scaler = GradScaler()
  for step, batch in enumerate(generator(), 1):
    # mixed precision training, cast to float16
    with autocast():
      loss = gpt.loss(batch[:, :-1].cuda(rank), batch[:, 1:].cuda(rank))
    
    if rank == 0:
      wandb.log({'loss': loss.item()})
      if step % 50 == 0:
        print(f'loss = {loss.item()}\tstep = {step}')
    
    # compute gradient for this process
    scaler.scale(loss).backward()
    
    # average gradients with rank 0 and broadcast
    for p in gpt.parameters():
      if p.requires_grad:
        # process i sends gradient to process 0 and gradients are summed
        dist.reduce(tensor=p.grad.data, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
          # process 0 averages gradient
          p.grad.data /= world_size
        # then sends it back to other processes
        dist.broadcast(tensor=p.grad.data, src=0)

    # since model in every process has same gradient, same update will occur 
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

  if rank == 0:
    wandb.finish()
    torch.save(gpt.state_dict(), 'model.pt')
 
  dist.destroy_process_group()  # destroy process once done


if __name__ == '__main__':
  os.environ['MASTER_ADDR'] = '127.0.0.1'  # IP address of process 0
  os.environ['MASTER_PORT'] = '10000'  # port open for process 0

  world_size = 4  # total number of processes 
  mp.spawn(process, args=(world_size,), nprocs=world_size)