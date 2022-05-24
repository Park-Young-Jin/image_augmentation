import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import preresnet as PRN
import PyramidNet as PYRM

from tensorboard_logger import configure, log_value

import pickle5 as pickle
import numpy as np
import pandas as pd
import math


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--net_type', default='PyramidNet', type=str,
                    help='networktype: resnet, resnext, densenet, pyamidnet, and so on')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model on ImageNet-1k dataset')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=int,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation for CIFAR datasets (default: True)')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--expname', default='PyramidNet', type=str,
                    help='name of experiment')
parser.add_argument('--RandomHorizontalFlip', default=0, type=int,
                    help='HorizontalFlip probablity')      
parser.add_argument('--RandomRotation', default=0, type=int,
                    help='RandomRotation degree')     
parser.add_argument('--RandomAffine', default=0, type=float,
                    help='RandomAffine translate')                                                               

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)
parser.set_defaults(augment=False)

best_err1 = 100
best_err5 = 100

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore

def main():
    global args, best_err1, best_err5
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.expname))

    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])    
        if args.augment:
        #    transform_train = transforms.Compose([
        #        transforms.RandomCrop(32, padding=4),
        #        transforms.RandomHorizontalFlip(),
        #        transforms.ToTensor(),
        #        normalize,
        #        ])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
        #transform_train_aug = transforms.Compose([transforms.RandomHorizontalFlip(p = 1),  transforms.ToTensor(), normalize, ])     
        transform_train_aug = transforms.Compose([transforms.RandomHorizontalFlip(p = args.RandomHorizontalFlip),transforms.RandomRotation(degrees = args.RandomRotation),transforms.RandomAffine(degrees = 0, translate =(args.RandomAffine,args.RandomAffine)),  transforms.ToTensor(), normalize, ])     

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True) 
            numberofclass = 100         
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset( 
                    [datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
                    datasets.CIFAR10('../data', train=True, download=True, transform=transform_train_aug)]
                ),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        elif args.dataset == 'cifar10_subset':
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset( 
                    [cifar10_subset(datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),2500),
                    cifar10_subset(datasets.CIFAR10('../data', train=True, download=True, transform=transform_train_aug),2500)]
                ),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10            
        else: 
            raise Exception ('unknown dataset: {}'.format(args.dataset)) 
    elif args.dataset == 'imagenet32':
        with open('/content/drive/MyDrive/imagenet/32/train_data', 'rb') as fo:
            d = pickle.load(fo)
        x = d['data']
        y = d['labels']    

        x_train  =  torch.FloatTensor(x)  
        y_train  =  torch.LongTensor(y)

        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        flip_yn = args.RandomHorizontalFlip
      
        x_augment, y_augment   =  augment(x,y,32,16,len(x),args.RandomHorizontalFlip,args.RandomRotation,args.RandomAffine)

        x_augment  =  torch.FloatTensor(x_augment.copy())  
        y_augment  =  torch.LongTensor(y_augment)

        
        dataset_augment = torch.utils.data.TensorDataset(x_augment, y_augment)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([dataset_augment,dataset]), batch_size=args.batch_size, shuffle=True)

        with open('/content/drive/MyDrive/imagenet/32/val_data', 'rb') as fo:
            d = pickle.load(fo)
        x = d['data']
        y = d['labels']    

        x_val  =  torch.FloatTensor(x)  
        y_val  =  torch.LongTensor(y)

        dataset_val = torch.utils.data.TensorDataset(x_val, y_val)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
        numberofclass = 10

    elif args.dataset == 'imagenet64':
        with open('/content/drive/MyDrive/imagenet/64/train_data', 'rb') as fo:
            d = pickle.load(fo)
        x = d['data']
        y = d['labels']    

        x_train  =  torch.FloatTensor(x)  
        y_train  =  torch.LongTensor(y)

        dataset = torch.utils.data.TensorDataset(x_train, y_train)

        x_augment, y_augment   =  augment(x,y,64,32,len(x),args.RandomHorizontalFlip,args.RandomRotation,args.RandomAffine)

        x_augment  =  torch.FloatTensor(x_augment.copy())  
        y_augment  =  torch.LongTensor(y_augment)

        dataset_augment = torch.utils.data.TensorDataset(x_augment, y_augment)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([dataset_augment,dataset]), batch_size=args.batch_size, shuffle=True)

        with open('/content/drive/MyDrive/imagenet/64/val_data', 'rb') as fo:
            d = pickle.load(fo)
        x = d['data']
        y = d['labels']    

        x_val  =  torch.FloatTensor(x)  
        y_val  =  torch.LongTensor(y)

        dataset_val = torch.utils.data.TensorDataset(x_val, y_val)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
        numberofclass = 10


    elif args.dataset == 'imagenet':
        traindir = os.path.join('args.data', 'train')
        valdir = os.path.join('args.data', 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    else: 
        raise Exception ('unknown dataset: {}'.format(args.dataset)) 

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.net_type))
        try:
            model = models.__dict__[str(args.net_type)](pretrained=True)
        except (KeyError, TypeError):
            print('unknown model')
            print('torchvision provides the follwoing pretrained model:', model_names)
            return
    else:
        print("=> creating model '{}'".format(args.net_type))
        if args.net_type == 'resnet':
            model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck) # for ResNet  
        elif args.net_type == 'preresnet':
            model = PRN.PreResNet(args.dataset, args.depth, numberofclass, args.bottleneck) # for Pre-activation ResNet  
        elif args.net_type == 'pyramidnet':
            model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass, args.bottleneck) # for ResNet  
        else:
            raise Exception ('unknown network architecture: {}'.format(args.net_type)) 


    if not args.distributed:
        if args.net_type.startswith('alexnet') or args.net_type.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            if torch.cuda.is_available() :
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = torch.nn.DataParallel(model)    
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_err1 = checkpoint['best_err1']
            best_err5 = checkpoint['best_err5']            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5
        print ('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        result_save = open("/content/drive/MyDrive/Colab Notebooks/졸업논문/실험결과/"+str(args.dataset)+"_"+str(args.RandomHorizontalFlip)+"_"+str(args.RandomRotation)+"_"+str(args.RandomAffine)+"_"+str(epoch)+"_"+str(best_err1)+"_"+str(err1)+".txt", 'w')
        result_save.close()
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    print ('Best accuracy (top-1 and 5 error):', best_err1, best_err5)  
 

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
            
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        
    
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                   epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_error', top1.avg, epoch)

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)

        # for PyTorch 0.3.x, use volatile=True for preventing memory leakage in evaluation phase:`
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        output = model(input_var)
        loss = criterion(output, target_var)

        # for PyTorch 0.4.x, volatile=True is replaced by with torch.no.grad(), so uncomment the followings:
        # with torch.no_grad():
        #     input_var = torch.autograd.Variable(input)
        #     target_var = torch.autograd.Variable(target)
        #     output = model(input_var)
        #     loss = criterion(output, target_var)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                   epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg, top5.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.expname) + 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar') or args.dataset == ('imagenet32'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs*0.5))) * (0.1 ** (epoch // (args.epochs*0.75)))
    elif args.dataset == ('imagenet64'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs*0.5))) * (0.1 ** (epoch // (args.epochs*0.75)))
    elif args.dataset == ('imagenet'):
        lr = args.lr * (0.1 ** (epoch // 30))
    
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #print(k)
        #print(batch_size)
        #print(correct)
        #print(correct[:k].view(-1))
        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        #print(correct_k)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

def augment(x,y,size,half_size,data_count,flip_yn,rotation_degree,affine_rate):
    
    
    if flip_yn == 1:
        x_return = x[:, :, :, ::-1]
    else :
        x_return = x    
    if rotation_degree != 0:
        rotation = []
        #회전 좌표 구하기
        for i in range(-half_size,half_size+1,1):
            for j in range(-half_size,half_size+1,1):
                if i*j !=0 :
                    #처음 좌표
                    coor_y = half_size-j
                    coor_x = i+half_size
                    if coor_y > half_size : coor_y = coor_y-1
                    if coor_x > half_size : coor_x = coor_x-1

                    #변경 좌표
                    coor_y_2 = half_size-(i*math.sin(math.pi * (rotation_degree / 180)) + j*math.cos(math.pi * (rotation_degree / 180)))
                    coor_x_2 = (i*math.cos(math.pi * (rotation_degree / 180)) - j*math.sin(math.pi * (rotation_degree / 180)))+half_size
                    if coor_y_2 > half_size : coor_y_2 = coor_y_2-1
                    if coor_x_2 > half_size : coor_x_2 = coor_x_2-1

                    rotation.append([coor_y,coor_x,coor_y_2,coor_x_2])
        data_rotation = pd.DataFrame(rotation)
        data_rotation = data_rotation[(data_rotation[2]>0) & (data_rotation[3]>0)&(data_rotation[2]<size) & (data_rotation[3]<size)]
        x_rotation = np.zeros((data_count,3,size,size))
        for i in range(size) :
            for j in range(size):
                data_rotation_2 = data_rotation[ ((round(data_rotation[2])==i)|(round(data_rotation[2],0)==i)) & ((round(data_rotation[3])==j)|(round(data_rotation[3],0)==j)) ]
                if len(data_rotation_2) > 0:
                    x_rotation[:,:,i,j] = np.mean(x_return[:,:,data_rotation_2[0],data_rotation_2[1]],axis=2)
        x_return = x_rotation

    if affine_rate != 0 :
        pixel = int(size*affine_rate)
        dummy = np.zeros((data_count,3,size,pixel))
        x_return = np.concatenate((dummy,x_return[:,:,:,:-pixel]),axis=3)
        dummy = np.zeros((data_count,3,pixel,size))
        x_return = np.concatenate((dummy,x_return[:,:,:-pixel,:]),axis=2)

    y_return = y
    return x_return,y_return

def cifar10_subset(trainset_in,cnt_num) :
    trainset_data = []
    trainset_label = []
    c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = 0,0,0,0,0,0,0,0,0,0

    for i in range(50000):
        c = trainset_in[i][1]
        #print(c)
        if c ==0 and c_0 < cnt_num: 
            c_0 += 1
            trainset_data.append(trainset_in[i][0])
            trainset_label.append(trainset_in[i][1])
        if c ==1 and c_1 < cnt_num: 
            c_1 += 1
            trainset_data.append(trainset_in[i][0])
            trainset_label.append(trainset_in[i][1])    
        if c ==2 and c_2 < cnt_num: 
            c_2 += 1
            trainset_data.append(trainset_in[i][0])
            trainset_label.append(trainset_in[i][1])    
        if c ==3 and c_3 < cnt_num: 
            c_3 += 1
            trainset_data.append(trainset_in[i][0])
            trainset_label.append(trainset_in[i][1])
        if c ==4 and c_4 < cnt_num: 
            c_4 += 1
            trainset_data.append(trainset_in[i][0])
            trainset_label.append(trainset_in[i][1])
        if c ==5 and c_5 < cnt_num: 
            c_5 += 1
            trainset_data.append(trainset_in[i][0])
            trainset_label.append(trainset_in[i][1])
        if c ==6 and c_6 < cnt_num: 
            c_6 += 1
            trainset_data.append(trainset_in[i][0])
            trainset_label.append(trainset_in[i][1])
        if c ==7 and c_7 < cnt_num: 
            c_7 += 1
            trainset_data.append(trainset_in[i][0])
            trainset_label.append(trainset_in[i][1])
        if c ==8 and c_8 < cnt_num: 
            c_8 += 1
            trainset_data.append(trainset_in[i][0])
            trainset_label.append(trainset_in[i][1])
        if c ==9 and c_9 < cnt_num: 
            c_9 += 1
            trainset_data.append(trainset_in[i][0])
            trainset_label.append(trainset_in[i][1])                     
        if c_0 >= cnt_num and c_1 >= cnt_num and c_2 >= cnt_num and c_3 >= cnt_num and c_4 >= cnt_num and c_5 >= cnt_num and c_6 >= cnt_num and c_7 >= cnt_num and c_8 >= cnt_num and c_9 >= cnt_num :
            break
    return torch.utils.data.TensorDataset(torch.stack(trainset_data), torch.LongTensor(trainset_label))

  


if __name__ == '__main__':
    main()
