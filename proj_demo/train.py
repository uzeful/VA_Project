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
from torch.optim.lr_scheduler import LambdaLR as LR_Policy

import models
from dataset import VideoFeatDataset as dset

from tools.glog_tools import get_logger
from tools.config_tools import Config
from tools import utils

from optparse import OptionParser
import pdb

parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="training configuration",
                  default="./configs/train_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)

mylog, logfile= get_logger(fileName=opt.log_name)
print(opt)
os.popen('cat {0} >> {1}'.format(opts.config, logfile))

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'models_checkpoint'

# make dir
if not os.path.exists(opt.checkpoint_folder):
    os.system('mkdir {0}'.format(opt.checkpoint_folder))

train_dataset = dset(opt.data_dir, flist=opt.flist)

mylog.info('number of train samples is: {0}'.format(len(train_dataset)))
mylog.info('finished loading data')

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

ngpu = int(opt.ngpu)
opt.manualSeed = random.randint(1, 10000) # fix seed
# opt.manualSeed = 123456

if torch.cuda.is_available() and not opt.cuda:
    mylog.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if ngpu == 1:
        mylog.info('so we use 1 gpu to training')
        mylog.info('setting gpu on gpuid {0}'.format(opt.gpu_id))

        if opt.cuda:
            torch.cuda.manual_seed(opt.manualSeed)

cudnn.benchmark = True
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
mylog.info('Random Seed: {0}'.format(opt.manualSeed))


# training function for metric learning
def train(train_loader, model, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # training mode
    model.train()

    end = time.time()
    for i, (vfeat, afeat) in enumerate(train_loader):
        #pdb.set_trace()
        # shuffling the index orders
        bz = vfeat.size()[0]
        orders = np.arange(bz)
        shuffle_orders = orders.copy()
        np.random.shuffle(shuffle_orders)

        # creating a new data with the shuffled indices
        afeat2 = afeat[torch.from_numpy(shuffle_orders)].clone()

        # concat the vfeat and afeat respectively
        afeat0 = torch.cat((afeat, afeat2), 0)
        vfeat0 = torch.cat((vfeat, vfeat), 0)

        # generating the labels
        # 1. the labels for the shuffled feats
        label1 = (orders == shuffle_orders + 0).astype('float32')
        target1 = torch.from_numpy(label1)

        # 2. the labels for the original feats
        label2 = label1.copy()
        label2[:] = 1
        target2 = torch.from_numpy(label2)

        # concat the labels together
        target = torch.cat((target2, target1), 0)
        target = 1 - target

        # put the data into Variable
        vfeat_var = Variable(vfeat0)
        afeat_var = Variable(afeat0)
        target_var = Variable(target)

        # if you have gpu, then shift data to GPU
        if opt.cuda:
            vfeat_var = vfeat_var.cuda()
            afeat_var = afeat_var.cuda()
            target_var = target_var.cuda()

        #pdb.set_trace()
        # forward, backward optimize
        sim = model(vfeat_var, afeat_var)   # inference simialrity
        loss = criterion(sim, target_var)   # compute contrastive loss

        ##############################
        # update loss in the loss meter
        ##############################
        losses.update(loss.data[0], vfeat0.size(0))

        ##############################
        # compute gradient and do sgd
        ##############################
        optimizer.zero_grad()
        loss.backward()

        ##############################
        # gradient clip stuff
        ##############################
        #utils.clip_gradient(optimizer, opt.gradient_clip)

        # update parameters
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses)
            mylog.info(log_str)


def main():
    global opt
    best_prec1 = 0
    # only used when we resume training from some checkpoint model
    resume_epoch = 0
    # train data loader
    # for loader, droplast by default is set to false
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                     shuffle=True, num_workers=int(opt.workers))

    # create model
    model = models.VAMetric()

    if not opt.train and opt.init_model != '':
        mylog.info('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    # Contrastive Loss
    criterion = models.ContrastiveLoss()

    if opt.cuda:
        mylog.info('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # adjust learning rate every lr_decay_epoch
    lambda_lr = lambda epoch: opt.lr_decay ** ((epoch + 1) // opt.lr_decay_epoch)   #poly policy
    scheduler = LR_Policy(optimizer, lambda_lr)

    for epoch in range(resume_epoch, opt.max_epochs):
    	#################################
        # train for one epoch
        #################################
        train(train_loader, model, criterion, optimizer, epoch, opt)
        scheduler.step()

        ##################################
        # save checkpoints
        ##################################

        # save model every 10 epochs
        if ((epoch+1) % opt.epoch_save) == 0:
            path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format(opt.checkpoint_folder, opt.prefix, epoch+1)
            utils.save_checkpoint(model.state_dict(), path_checkpoint)

if __name__ == '__main__':
    main()
