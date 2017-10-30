#!/usr/bin/env python

from __future__ import print_function
import argparse
import random
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel # for multi-GPU training
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import LambdaLR as LR_Policy

import model2 as models
from dataset2 import VideoFeatDataset as dset
import utils

from visdom_tools import VisdomPlotter as Dashboard
from glog_tools import get_logger
from config_tools import Config
from optparse import OptionParser
import pdb

parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="net configuration",
                  default="./test_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)

mylog, logfile= get_logger(fileName=opt.log_name)
print(opt)
os.popen('cat {0} >> {1}'.format(opts.config, logfile))

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'

test_dataset = dset(opt.data_dir, opt.flist)

mylog.info('number of test samples is: {0}'.format(len(test_dataset)))
mylog.info('finished loading data')

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

ngpu = int(opt.ngpu)
if torch.cuda.is_available() and not opt.cuda:
    mylog.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if ngpu == 1:
        mylog.info('so we use gpu 1 for testing')
        mylog.info('setting gpu on gpuid {0}'.format(opt.gpu_id))

cudnn.benchmark = True

# test function for metric learning
def test(test_loader, model, opt):
    """
    train for one epoch on the training set
    """
    # training mode
    model.eval()

    end = time.time()
    sim_mat = []
    right = 0
    for _, (vfeat, _) in enumerate(test_loader):
        for _, (_, afeat) in enumerate(test_loader):
            #pdb.set_trace()
            # shuffling the index orders
            bz = vfeat.size()[0]
            for k in np.arange(bz):
                cur_vfeat = vfeat[k].clone()
                cur_vfeats = cur_vfeat.repeat(bz, 1, 1)

                vfeat_var = Variable(cur_vfeats)
                afeat_var = Variable(afeat)

                if opt.cuda:
                    vfeat_var = vfeat_var.cuda()
                    afeat_var = afeat_var.cuda()

                cur_sim = model(vfeat_var, afeat_var)
                if k == 0:
                    simmat = cur_sim.clone()
                else:
                    simmat = torch.cat((simmat, cur_sim), 1)
            sorted, indices = torch.sort(simmat, 0)
            np_indices = indices.cpu().data.numpy()
            topk = np_indices[:opt.topk,:]
            for k in np.arange(bz):
                order = topk[:,k]
                if k in order:
                    right = right + 1
            print(simmat)
            print('accuracy (top{}): {:.3}'.format(opt.topk, right/bz))

def main():
    global opt
    # test data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                     shuffle=False, num_workers=int(opt.workers))

    # create model
    model = models.VAMetric()

    if opt.init_model != '':
        mylog.info('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    if opt.cuda:
        mylog.info('shift model to GPU .. ')
        model = model.cuda()

    test(test_loader, model, opt)


if __name__ == '__main__':
    main()
