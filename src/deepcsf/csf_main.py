"""
The training/testing routines for contrast sensitivity functions.
"""

import numpy as np
import argparse
import os

import torch
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter

from . import models
from . import dataloader
from . import utils


def setup_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs of training")
    parser.add_argument("--initial_epoch", type=int, default=0, help="the staring epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--train_samples", type=int, default=500,
                        help="Number of train samples at each epoch")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of CPU workers")
    parser.add_argument("--lr", type=float, default=0.1, help="SGD: learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD: momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="SGD: weight decay")
    parser.add_argument("--out_dir", type=str, default="./csf_out/", help="the output directory")
    parser.add_argument("--test_net", type=str, default=None, help="the path to test network")
    parser.add_argument("--resume", type=str, default=None, help="the path to training checkpoint")
    return parser.parse_args(argv)


def main(argv):
    args = setup_arguments(argv)
    # creating the output dir if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # setting variables that are constant in our project
    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]
    args.target_size = 224
    # logging the arguments in the output folder
    utils.save_arguments(args)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.criterion = torch.nn.CrossEntropyLoss().to(args.device)
    # opening the tensorboard
    args.tb_writers = dict()
    for mode in ['train', 'test']:
        args.tb_writers[mode] = SummaryWriter(os.path.join(args.out_dir, mode))

    # making the network
    network = models.LinearProbe(args.device)
    network = network.to(args.device)
    if args.test_net is None:  # training
        _train_contrast_discriminator(network, args)
    else:  # testing
        _test_csf(network, args)

    # closing the tensorboard
    for mode in args.tb_writers.keys():
        args.tb_writers[mode].close()


def _epoch_loop(network, db_loader, optimiser, args, step, name_gen=None):
    # usually the code for train/test has a large overlap.
    is_train = False if optimiser is None else True

    # model should be in train/eval model accordingly
    network.train() if is_train else network.eval()
    epoch_type = 'train' if is_train else 'test'
    tb_writer = args.tb_writers[epoch_type]

    accuracies = []
    losses = []
    outputs = []
    with torch.set_grad_enabled(is_train):
        for batch_ind, (img1, img2, target) in enumerate(db_loader):
            # moving the image and GT to device
            img1 = img1.to(args.device)
            img2 = img2.to(args.device)
            target = target.to(args.device)

            # writing to tensorboard
            if batch_ind == 0:
                img_disp = torch.cat([img1, img2], dim=3)
                # normalising the images
                for i in range(img_disp.shape[1]):
                    img_disp[:, i, ] = (img_disp[:, i, ] * args.std[i]) + args.mean[i]
                img_disp = torchvision.utils.make_grid(img_disp, nrow=4)
                img_name = name_gen() if name_gen is not None else '%s_batch' % epoch_type
                tb_writer.add_image('{}'.format(img_name), img_disp, step)

            # calling the network
            output = network(img1, img2)
            outputs.extend([out for out in output])

            # computing the loss function
            loss = args.criterion(output, target)
            losses.extend([loss.item() for _ in range(img1.size(0))])
            # computing the accuracy
            acc = utils.accuracy(output, target)[0].cpu().numpy()
            accuracies.extend([acc for _ in range(img1.size(0))])

            if is_train:
                # compute gradient and do SGD step
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
    return accuracies, losses, outputs


def _train_contrast_discriminator(network, args):
    train_logs = {'acc': [], 'loss': []}

    # optimiser
    params_to_optimize = [{'params': [p for p in network.fc.parameters()]}]
    optimizer = torch.optim.SGD(
        params_to_optimize, lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    # if resuming a previously training process
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        network.load_state_dict(checkpoint['network']['state_dict'])
        args.initial_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_logs = checkpoint['train_logs']
    network = network.to(args.device)

    # making the train loader
    train_loader = dataloader.train_dataloader(args)

    # doing epoch
    tb_writer = args.tb_writers['train']
    for epoch in range(args.initial_epoch, args.epochs):
        e_log = _epoch_loop(network, train_loader, optimizer, args, epoch)

        print('[%.3d] loss=%.4f\tacc=%0.2f' % (epoch, np.mean(e_log[1]), np.mean(e_log[0])))
        train_logs['acc'].append(np.mean(e_log[0]))
        train_logs['loss'].append(np.mean(e_log[1]))

        # writing to tensorboard
        tb_writer.add_scalar("{}".format('loss'), train_logs['loss'][-1], epoch)
        tb_writer.add_scalar("{}".format('accuracy'), train_logs['acc'][-1], epoch)

        # saving model progress
        np.savetxt(
            '%s/progress.csv' % args.out_dir,
            np.stack([train_logs['acc'], train_logs['loss']]).T,
            delimiter=',', header='accuracy,loss'
        )

        # saving the checkpoint
        # all variables that are necessary to resume the training must be saved at this point
        # a non-exhaustive list include network, optimizer, scheduler, normalisation mean/std, etc.
        utils.save_checkpoint(
            {
                'epoch': epoch,
                'network': {
                    'arch': 'resnet18',
                    'layer': 8,
                    'state_dict': network.state_dict()
                },
                'preprocessing': {'mean': args.mean, 'std': args.std},
                'optimizer': optimizer.state_dict(),
                'target_size': args.target_size,
                'train_logs': train_logs
            },
            args.out_dir
        )
    return


def _test_csf(network, args):
    checkpoint = torch.load(args.test_net, map_location='cpu')
    network.load_state_dict(checkpoint['network']['state_dict'])
    network = network.to(args.device)
    network.eval()

    # spatial frequencies
    sfs = [i for i in range(1, int(args.target_size / 2) + 1) if args.target_size % i == 0]

    tb_writer = args.tb_writers['test']
    sensitivities = []
    for i in range(len(sfs)):
        print('testing spatial frequency', sfs[i])
        res_i = _sensitivity_sf(network, sfs[i], args)
        sensitivities.append(1 / res_i[-1][-1])
        tb_writer.add_scalar("{}".format('csf'), sensitivities[-1], sfs[i])
    np.savetxt(
        '%s/csf.csv' % args.out_dir,
        np.stack([sfs, sensitivities]).T,
        delimiter=',', header='spatial_frequency,sensitivity'
    )


def _sensitivity_sf(network, sf, args):
    """Computing the psychometric function for given spatial frequency."""
    low = 0
    high = 1
    mid = (low + high) / 2

    res_sf = []
    attempt_i = 0

    def name_gen():
        return 'sf_%.3d_batch' % sf

    # th=0.749 (instead of standard psychometric function of 0.75) because of small test dataset
    th = 0.749
    while True:
        db_loader = dataloader.test_dataloader(args, sf, contrast=mid)
        val_log = _epoch_loop(network, db_loader, None, args, attempt_i, name_gen)
        accuracy = np.mean(val_log[0]) / 100
        res_sf.append(np.array([sf, accuracy, mid]))
        new_low, new_mid, new_high = utils.staircase_paradigm(accuracy, low, mid, high, th=th)
        if new_mid is None or attempt_i == 20:
            break
        else:
            low, mid, high = new_low, new_mid, new_high
        attempt_i += 1
    return res_sf
