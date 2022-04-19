import os
import DataProvider_HiFA_train
import HFANet

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import Losses
import torch.nn.functional as F
import numpy as np

BATCHSIZE = 128
EPOCHS = 101

parser = argparse.ArgumentParser()
parser.add_argument(
    "--nodes", default=1, type=int, help="number of nodes for distributed training"
)
parser.add_argument(
    "--ngpus_per_node",
    default=4,
    type=int,
    help="number of GPUs per node for distributed training",
)
parser.add_argument(
    "--dist-url",
    default="tcp://127.0.0.1:12306",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--node_rank", default=0, type=int, help="node rank for distributed training"
)
parser.add_argument(
    '--save_model_dir', type=str, default='trained_model/', help='the model save directory')

parser.add_argument(
    "--lr", default=0.05, type=float, help='model initial learning rate'
)

def main():
    args = parser.parse_args()
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    # global_world_size = ngpus_per_node * nodes
    args.global_world_size = args.ngpus_per_node * args.nodes
    mp.spawn(train_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.2 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_worker(local_rank, ngpus_per_node, args):
    args.global_rank = args.node_rank*ngpus_per_node+local_rank
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.global_world_size,
        rank=args.global_rank,
    )
    device = torch.device(type="cuda", index=local_rank)
    torch.cuda.set_device(local_rank)
    print(f"[init] == local rank: {local_rank}, global rank: {args.global_rank} ==\n")

    net = HFANet.HFANet()
    net.to(device)
    net = DDP(net,
              device_ids=[local_rank],
              output_device=local_rank)

    trainset = DataProvider_HiFA_train.HiFADataset()

    train_sampler = torch.utils.data.distributed.DistributedSampler( trainset, shuffle=True )
    train_loader = torch.utils.data.DataLoader( trainset,
                                                batch_size=BATCHSIZE,
                                                num_workers=16,
                                                pin_memory=True,
                                                sampler=train_sampler)

    hifa_loss = Losses.Losses()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    if args.global_rank == 0:
        print(" =======  Training  ======= \n")

    net.train()
    for ep in range(1, EPOCHS + 1):
        train_sampler.set_epoch(ep)
        lr_scheduler.step()
        running_loss = 0.
        for idx, (input_feats, gtlabel) in enumerate(train_loader):
            input_feats = input_feats.to(device)
            gtlabel = gtlabel.to(device)
            logits = net(input_feats)

            loss = hifa_loss.compute_loss(logits=logits, gtlabel=gtlabel)
            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            current_lr = optimizer.param_groups[0]['lr']


            if args.global_rank == 0 and ((idx + 1) % 25 == 0 or (idx + 1) == len(train_loader)):
                print("== step: [{:3}/{}] [{}/{}] | loss: {:.3f}, lr={}".format(idx + 1,
                                                                         len(train_loader),
                                                                         ep,
                                                                         EPOCHS,
                                                                         loss / (idx + 1),
                                                                         current_lr))

            if args.global_rank == 0 and ep%10 == 0 and ep > 50:
                model_save_basename = 'model_epoch_{}.pth'.format(ep)
                outdict = {
                    'epoch': ep,
                    'model': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save( outdict, os.path.join(args.save_model_dir, model_save_basename))



    if args.global_rank == 0:
        print("\n=======  Training Finished  ======= \n")


if __name__ == '__main__':
    main()
