# GPT training using distributed data parallel
# load text file and tokenize using youtokentome
# or load PyTorch tensor of tokens
# loads GPT model using YAML config

import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from gpt import *
from lion_pytorch import Lion
from torch.cuda.amp import GradScaler, autocast
from argparse import ArgumentParser
from yaml import safe_load
from torch.distributed import (
  init_process_group, destroy_process_group, reduce, broadcast, ReduceOp
)


def process(rank, world, config, dataloader):
  init_process_group(backend='nccl', rank=rank, world=world)
  print(f'started process {rank}')
  
  model = GPT(*config.values()).cuda(rank)
  optimizer = Lion(model.parameters(), lr=1e-4)
  
  scaler = GradScaler()
  
  for i, (x, y) in enumerate(dataloader(), 1):
    with autocast():
      loss = model.loss(x.cuda(rank), y.cuda(rank))
    
    loss = scaler.scale(loss)   
    if rank == 0:
      wandb.log({'loss': loss.item()}) 
    loss.backward()

    if world > 1: 
      for p in model.parameters():
        reduce(tensor=p.grad, op=ReduceOp.SUM, dst=0)
        if rank == 0:
          p.grad /= world
        broadcast(tensor=p.grad, src=0)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

  if rank == 0:
    torch.save(model.state_dict(), 'model.pt')
    print('model saved')
  
  destroy_process_group()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--project', type=str)
  parser.add_argument('--run', type=str)
  parser.add_argument('--addr', type=str, default='127.0.0.1')
  parser.add_argument('--port', type=str, default='10000')
  parser.add_argument('--corpus', type=str)
  parser.add_argument('--config', type=str) 
  parser.add_argument('--tokenizer', type=str, default=None)
  parser.add_argument('--tokens', type=str, default=None)
  parser.add_argument('--batch_size', type=int)
  parser.add_argument('--world', type=int)
  args = parser.parse_args() 

  # IP address and port of master worker
  os.environ['MASTER_ADDR'] = args.addr 
  os.environ['MASTER_PORT'] = args.port

  # init wandb project and run
  wandb.init(project=args.project)
  wandb.run.name = args.run 

  # load config from YAML
  config = safe_load(open(args.config, 'r'))

  # load tokens
  if args.corpus and args.tokenizer:
    text = open(args.corpus, 'r')
    bpe = yttm.BPE(args.tokenizer)
    assert config['v'] == bpe.vocab_size(), \
      "tokenizer and config vocab sizes don't match"
    tokens = bpe.encode(text)
  elif args.tokens:
    tokens = torch.load(args.tokens)
  else:
    raise RuntimeError(
      'either tokens path should be given or both tokenizer and corpus paths'
    )
  
  split = len(tokens) // args.world  # number of GPUs
  ids = torch.tensor(tokens)[:args.world * split].split(split)

  def dataloader():
    for i in range(args.iterations):
      indices = random.choice(len(ids) - args.l - 1, k=args.batch_size)
      batch = torch.stack([ids[i:i + l + 1] for i in indices])
      yield batch[:, :-1], batch[:, 1:]

  # start processes
  mp.spawn(process, args=(args.world, config, dataloader), nprocs=args.world)
