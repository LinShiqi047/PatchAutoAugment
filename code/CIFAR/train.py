import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

# ***** # 
# import augmentation policy
# ***** # 
from augs.autoaugment import CIFAR10Policy
from augs.cutout import Cutout

# ***** # 
# import model
# ***** # 
# WRN
from model.wideresnet import WideResNet
# SS(shake-shake): 2 options
from model.shake_shake import ShakeShake
from model.ss_resnet import ShakeResNet
# SD(shakedrop)
from model.shake_pyramidnet import ShakePyramidNet 

# ***** # 
# import PAA module
# ***** # 
from model.PAA import *   
from model.A2Cmodel import ActorCritic

# used for logging to TensorBoard
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
# default
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    help='print frequency (default: 50)')

# FRY
parser.add_argument('--model', default='WRN', type=str, choices=['WRN', 'SS', 'SD'],
                    help='wideresnet, shake-shake, shakedrop')
parser.add_argument('--SS_w_base', default=32, type=int, choices=[32, 96, 112],
                    help='only used when model as SS')
parser.add_argument('--aug', default='base', type=str, choices=['base', 'cutout', 'AA', 'onlyAA', 'PAA'],
                    help='augmentation type')
parser.add_argument('--is_leinao', default=False, action='store_true')
parser.set_defaults(augment=True)
best_prec1 = 0

def get_hyper_params(args):
    hyper_params = {}
    print('Dataset: {}'.format(args.dataset))
    print('Mainnet: {}'.format(args.model))
    if args.dataset == 'cifar10':
        if args.model == 'WRN': 
            hyper_params['lr'] = 0.1
            hyper_params['WD'] = 5e-4
            hyper_params['epochs'] = 200
            hyper_params['nesterov'] = True
            hyper_params['momentum'] = 0.9
        elif args.model == 'SS':
            hyper_params['lr'] = 0.2
            hyper_params['WD'] = 1e-4
            hyper_params['epochs'] = 600
            hyper_params['nesterov'] = True
            hyper_params['momentum'] = 0.9
        elif args.model == 'SD':
            hyper_params['lr'] = 0.1
            hyper_params['WD'] = 1e-4
            hyper_params['epochs'] = 600
            hyper_params['nesterov'] = True
            hyper_params['momentum'] = 0.9
        else:
            raise('invalid model !')
    elif args.dataset == 'cifar100':
        if args.model == 'WRN':
            hyper_params['lr'] = 0.1
            hyper_params['WD'] = 5e-4
            hyper_params['epochs'] = 200
            hyper_params['nesterov'] = True
            hyper_params['momentum'] = 0.9
        elif args.model == 'SS':
            hyper_params['lr'] = 0.1
            hyper_params['WD'] = 5e-4
            hyper_params['epochs'] = 1200
            hyper_params['nesterov'] = True
            hyper_params['momentum'] = 0.9
        elif args.model == 'SD':
            hyper_params['lr'] = 0.5
            hyper_params['WD'] = 1e-4
            hyper_params['epochs'] = 1200
            hyper_params['nesterov'] = True
            hyper_params['momentum'] = 0.9
        else:
            raise('invalid model !')    
    else:
        raise('invalid dataset !')

    if args.aug == 'PAA':
        hyper_params['lr_a2c'] = 1e-4
    
    return hyper_params

def build_model(args, hyper_params):
    num_classes = 10 if args.dataset == 'cifar10' else 100
    if args.model == 'WRN':
        model = WideResNet(depth=28, num_classes=num_classes,
                                widen_factor=10, dropRate=0.0)
    elif args.model == 'SS':
        model_config = OrderedDict([
        ('arch', 'shake_shake'),
        # ('depth', args.depth),
        ('depth', 26),
        ('base_channels', args.SS_w_base),
        ('shake_forward', True),
        ('shake_backward', True),
        ('shake_image', True),
        ('input_shape', (1, 3, 32, 32)),
        ('n_classes', num_classes),
        ])
        config = OrderedDict([
        ('model_config', model_config),
        ])
        model = ShakeShake(num_classes=num_classes, depth=26, base_channels=args.SS_w_base)

        # model = ShakeResNet(depth=26, w_base=args.SS_w_base, classes=num_classes)
    elif args.model == 'SD':
        model = ShakePyramidNet(depth=110, alpha=270, label=num_classes)

    return model
    
def build_dataloader(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    print('Augmentation: {}'.format(args.aug))
    if args.aug == 'base':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                (4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    elif args.aug == 'cutout':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                            (4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            # CIFAR10Policy(), 
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            normalize,
            ])
    elif args.aug == 'AA':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                            (4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(), 
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            normalize,
            ])
    elif args.aug == 'onlyAA':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                            (4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(), 
            transforms.ToTensor(),
            # Cutout(n_holes=1, length=16),
            normalize,
            ])
    elif args.aug == 'PAA':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                            (4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            # CIFAR10Policy(), 
            transforms.ToTensor(),
            # Cutout(n_holes=1, length=16),
            # normalize,
            ])
        pass

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    if not args.is_leinao:
        path_datasets = '../data'
    else:
        path_datasets = "/data/bitahub/CIFAR/{}".format(args.dataset) 
    kwargs = {'num_workers': 4, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](path_datasets, train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](path_datasets, train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    return train_loader, val_loader

def build_f_output(args):
    if args.is_leinao:
        output_file = '/output/output_{}_{}'.format(args.dataset, args.model)
    else:
        output_file = 'output_/{}/{}'.format(args.dataset, args.model)
    if not os.path.exists(output_file):
        os.makedirs(output_file, exist_ok=True)
    
    f_output = open(os.path.join(output_file, '{}.txt'.format(args.aug)), mode='w')

    return f_output

def main():
    global args, best_prec1
    args = parser.parse_args()
    hyper_params = get_hyper_params(args)
    f_output = build_f_output(args)
    args.name = args.model

    # ****** # 
    # dataloader
    # ****** # 
    train_loader, val_loader = build_dataloader(args)

    # ****** # 
    # create model, criterion, optimizer ...
    # ****** # 
    model = build_model(args, hyper_params).cuda()
    # model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), hyper_params['lr'],
                                momentum=hyper_params['momentum'], nesterov=hyper_params['nesterov'],
                                weight_decay=hyper_params['WD'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)* hyper_params['epochs'])
    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    if args.aug == 'PAA':
        model_a2c = ActorCritic(len(ops)).cuda()
        optimizer_a2c = torch.optim.SGD(model_a2c.parameters(), hyper_params['lr_a2c'], 
                                    momentum=0.9, nesterov=True,
                                    weight_decay=1e-4)
        scheduler_a2c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a2c, len(train_loader)* hyper_params['epochs'])

    cudnn.benchmark = True
    for epoch in range(0, hyper_params['epochs']):
        # train and evaluate for one epoch
        if args.aug == 'PAA':
            train(train_loader, model, criterion, optimizer, scheduler, epoch, f_output,
                model_a2c, optimizer_a2c, scheduler_a2c)
        else:
            train(train_loader, model, criterion, optimizer, scheduler, epoch, f_output)
        prec1 = validate(val_loader, model, criterion, epoch, f_output)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)
    f_output.write('Best accuracy: {}'.format(best_prec1)+'\n')

def train(train_loader, model, criterion, optimizer, scheduler, epoch, f_output,
        model_a2c=None, optimizer_a2c=None, scheduler_a2c=None):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()

    if args.aug == 'PAA':
        losses_a2c = AverageMeter()
        model_a2c.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        if args.aug == 'PAA':
            # patch data augmentation
            operations ,operation_logprob , operation_entropy , state_value = \
                rl_choose_action(input, model_a2c)
            input_aug = patch_auto_augment(input, operations, args.batch_size)

            output = model(input_aug.detach())
            loss = criterion(output, target)

            # PAA loss
            reward = loss.detach()
            loss_a2c = a2closs(operation_logprob, operation_entropy, state_value, reward)

            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            losses_a2c.update(loss_a2c.data.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer_a2c.zero_grad()
            loss_a2c.backward()
            optimizer_a2c.step()
            scheduler_a2c.step()
        else:
            output = model(input)
            loss = criterion(output, target)

            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.aug == 'PAA':
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'A2CLoss {loss_a2c.val:.4f} ({loss_a2c.avg:.4f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        loss=losses, top1=top1, loss_a2c=losses_a2c))
                f_output.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'A2CLoss {loss_a2c.val:.4f} ({loss_a2c.avg:.4f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        loss=losses, top1=top1, loss_a2c=losses_a2c) +'\n')
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        loss=losses, top1=top1))
                f_output.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        loss=losses, top1=top1) +'\n')

def validate(val_loader, model, criterion, epoch, f_output):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
            f_output.write('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1)+'\n')

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    f_output.write(' * Prec@1 {top1.avg:.3f}'.format(top1=top1)+'\n')
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

    # ? for debug
    # global args
    # args = parser.parse_args()

    # hyper_params = get_hyper_params(args)
    # print(hyper_params)
