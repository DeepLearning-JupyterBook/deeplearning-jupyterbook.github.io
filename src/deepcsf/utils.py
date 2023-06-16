"""
A list of utility functions.

This module often contain a list of functions that are useful for several routines.
The functions in this module are often public and accessible to all other modules.

If this module becomes too completed (too many functions) it can be easily broken down into
several utility modules (with an appropriate name) specialised to specific routines.
"""

import os
import json

import torch


def staircase_paradigm(acc, low, mid, high, th, ep=1e-4):
    """Staircase paradigm, gradually changing the stimuli until accuracy reaches the threshold"""
    diff_acc = acc - th
    if abs(diff_acc) < ep:
        return None, None, None
    elif diff_acc > 0:
        new_mid = (low + mid) / 2
        return low, new_mid, mid
    else:
        new_mid = (high + mid) / 2
        return mid, new_mid, high


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_arguments(args):
    """Dumping all arguments in a JSON file"""
    json_file_name = os.path.join(args.out_dir, 'args.json')
    with open(json_file_name, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)


def save_checkpoint(state, out_dir, filename='checkpoint.pth.tar'):
    """Saving the network checkpoint"""
    file_path = os.path.join(out_dir, filename)
    torch.save(state, file_path)
