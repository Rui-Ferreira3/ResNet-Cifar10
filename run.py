import argparse
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
from tqdm import tqdm
import resnet

HIST_NAME = f'histograms/hist.png'
default_device = "cpu"

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--train', dest='train', action='store_true',
                    help='Trained model')
parser.add_argument('--hist', dest='hist', action='store_true',
                    help='Save conv weight/input/output histograms as PNG after each epoch')
parser.add_argument('-m', '--model', dest='model',
                    help='The filename of the trained model',
                    default=None, type=str)
parser.add_argument('--sim', dest='sim', action='store_true',
                    help='Use accelerator simulator for inference')
best_prec1 = 0
pretrained = True

def run(cli_args=None, conv2d=None):
    global args, best_prec1, pretrained
    args = parser.parse_args(cli_args)

    if args.model is not None:
        pretrained = False

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = resnet.__dict__[args.arch]()
    if args.sim:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../IdxedAccum/sw'))
        import AxCConv
        device = "cpu"
    else:
        device = torch.device(default_device)
        model = torch.nn.DataParallel(model) if default_device == "cuda" else model

    model.to(device)

    if conv2d is not None:
        print("Replacing conv2d to use hw")
        model = replace_conv2d(model, conv2d)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # Check if in test mode
    if not args.train:
        # Check if pre trained module exists
        if pretrained:
            model_path = f"./pretrained_models/{args.arch}.th"
        else:
            model_path = args.model
        assert os.path.isfile(model_path), f"Model '{model_path}' does not exist"
        if args.sim:
            print(f"Testing medel '{model_path}' on simulator")
        else:
            print(f"Testing medel '{model_path}' on {default_device}")

        collector = resnet.ConvDataCollector() if args.hist else None

        trained_model = torch.load(model_path, weights_only=True, map_location=device)
        if args.sim or not isinstance(model, torch.nn.DataParallel):
            state_dict = {k.replace('module.', ''): v for k, v in trained_model['state_dict'].items()}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(trained_model['state_dict'])
        model.training = False

        start = time.time()
        top1, collector = validate(val_loader, model, criterion, collector=collector)
        t = time.time() - start
        print(f"Total inference time: {t:.3f} seconds")

        return t, top1, collector

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1, _ = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = input.to(device)
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, collector=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    if args.sim:
        device = "cpu"
        mydevice = torch.device("axcconvdevice")
        hooks = register_device_hooks(model, mydevice)
    else:
        device = torch.device(default_device)

    running_top1_corrects = 0
    processed_data = 0
    pbar = tqdm(val_loader, desc='Validating')
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(pbar):
            labels = labels.to(device)
            inputs = inputs.to(device)

            # enable collector on first batch only
            if collector is not None and i == 0:
                resnet.set_collector(collector)

            # compute output
            outputs = model(inputs)

            if collector is not None and i == 0:
                resnet.set_collector(None)

            loss = criterion(outputs, labels)

            outputs = outputs.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, labels)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            processed_data += inputs.size(0)

            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'top1': f'{top1.avg:.2f}%',
            })


    if args.sim:
        for h in hooks:
            h.remove()

    return top1.avg, collector

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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

def register_device_hooks(model, device):
    hooks = []
    conv_modules = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]

    def make_hooks(mod):
        def pre_hook(mod, inputs):
            x = inputs[0]
            if mod.weight.device.type != device:
                mod.weight = torch.nn.Parameter(mod.weight.detach().to(device), requires_grad=False)
                if mod.bias is not None:
                    mod.bias = torch.nn.Parameter(mod.bias.detach().to(device), requires_grad=False)
            return (x.to(device),)

        def post_hook(_mod, _inputs, output):
            return output.to('cpu')

        return pre_hook, post_hook

    for module in conv_modules:
        pre, post = make_hooks(module)
        hooks.append(module.register_forward_pre_hook(pre))
        hooks.append(module.register_forward_hook(post))

    return hooks

def replace_conv2d(model, conv2d_cls):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and not isinstance(module, conv2d_cls):
            new_conv = conv2d_cls(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            new_conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_conv.bias.data = module.bias.data.clone()

            setattr(model, name, new_conv)
        else:
            # recurse into children (handles nested Sequential, BasicBlock, etc.)
            replace_conv2d(module, conv2d_cls)
    return model

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    _, top1_sim, collector_sim = run()

    if args.hist:
        args.sim = False

        _, top1_float, collector_float = run()

        os.makedirs('histograms', exist_ok=True)
        if collector_sim is not None and collector_float is not None:
            collector_sim.plot_histograms_overlay(
                collector_float,
                save_path=HIST_NAME,
                title=f'Sim Top-1: {top1_sim:.3f}%  vs  Float Top-1: {top1_float:.3f}%',
            )
        elif collector_sim is not None:
            collector_sim.plot_histograms(save_path=HIST_NAME, title=f'Sim Top-1: {top1_sim:.3f}%')
