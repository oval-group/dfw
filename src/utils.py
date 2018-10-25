import os
import sys
import socket
import torch
import logger
import random
import numpy as np


def regularization(model, l2):
    reg = 0.5 * l2 * sum([p.data.norm() ** 2 for p in model.parameters()]) if l2 else 0
    return reg


def set_seed(args, print_out=True):
    if args.seed is None:
        np.random.seed(None)
        args.seed = np.random.randint(1e5)
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


def get_xp(args, model, optimizer):

    # various useful information to store
    args.command_line = 'python ' + ' '.join(sys.argv)
    args.pid = os.getpid()
    args.cwd = os.getcwd()
    args.hostname = socket.gethostname()

    xp = logger.Experiment(args.xp_name,
                           use_visdom=args.visdom,
                           visdom_opts={'server': args.server,
                                        'port': args.port},
                           time_indexing=False, xlabel='Epoch')

    xp.SumMetric(name='epoch', to_plot=False)

    xp.AvgMetric(name='acc', tag='train')
    xp.AvgMetric(name='acc', tag='val')
    xp.AvgMetric(name='acc', tag='test')
    xp.BestMetric(name='acc', tag='valbest')

    xp.TimeMetric(name='timer', tag='train')
    xp.TimeMetric(name='timer', tag='val')
    xp.TimeMetric(name='timer', tag='test')

    xp.AvgMetric(name='loss', tag='train')
    xp.AvgMetric(name='obj', tag='train')
    xp.AvgMetric(name='reg')

    xp.log_config(vars(args))

    xp.AvgMetric(name="gamma")
    xp.SimpleMetric(name='eta')

    if args.log:
        # log at each epoch
        xp.Epoch.add_hook(lambda: xp.to_json('{}/results.json'.format(xp.name_and_dir)))
        # log after final evaluation on test set
        xp.Acc_Test.add_hook(lambda: xp.to_json('{}/results.json'.format(xp.name_and_dir)))
        # save with different names at each epoch if needed
        if args.dump_epoch:
            filename = lambda: '{}-{}/model.pkl'.format(xp.name_and_dir, int(xp.Epoch.value))
        else:
            filename = lambda: '{}/model.pkl'.format(xp.name_and_dir)
        xp.Epoch.add_hook(lambda: save_state(model, optimizer, filename()))

        # save results and model for best validation performance
        xp.Acc_Valbest.add_hook(lambda: xp.to_json('{}/best_results.json'.format(xp.name_and_dir)))
        xp.Acc_Valbest.add_hook(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(xp.name_and_dir)))

    return xp


def save_state(model, optimizer, filename):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)


@torch.autograd.no_grad()
def accuracy(out, targets, topk=1):
    if topk == 1:
        _, pred = torch.max(out, 1)
        acc = torch.mean(torch.eq(pred, targets).float())
    else:
        _, pred = out.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        acc = correct[:topk].view(-1).float().sum(0) / out.size(0)

    return 100. * acc


def update_metrics(xp, state):
    xp.Acc_Train.update(state['acc'], n=state['size'])
    xp.Loss_Train.update(state['loss'], n=state['size'])
    xp.Gamma.update(state['gamma'], n=state['size'])


def log_metrics(xp):

    # Average of accuracy and loss on training set
    xp.Acc_Train.log_and_reset()
    xp.Loss_Train.log_and_reset()
    xp.Obj_Train.log_and_reset()
    xp.Reg.log_and_reset()

    # timer of epoch
    xp.Timer_Train.log_and_reset()

    # Log step-size
    xp.Gamma.log_and_reset()
    xp.Eta.log_and_reset()
