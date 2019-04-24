import os
import sys
import socket
import torch
import mlogger
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


def setup_xp(args, model, optimizer):

    env_name = args.xp_name.split('/')[-1]
    if args.visdom:
        plotter = mlogger.VisdomPlotter({'env': env_name, 'server': args.server, 'port': args.port})
    else:
        plotter = None

    xp = mlogger.Container()

    xp.config = mlogger.Config(plotter=plotter, **vars(args))

    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="training")
    xp.train.loss = mlogger.metric.Average(plotter=plotter, plot_title="Objective", plot_legend="loss")
    xp.train.obj = mlogger.metric.Simple(plotter=plotter, plot_title="Objective", plot_legend="objective")
    xp.train.reg = mlogger.metric.Simple(plotter=plotter, plot_title="Objective", plot_legend="regularization")
    xp.train.weight_norm = mlogger.metric.Simple(plotter=plotter, plot_title="Weight-Norm")
    xp.train.gamma = mlogger.metric.Average(plotter=plotter, plot_title="Gamma")
    xp.train.eta = mlogger.metric.Simple(plotter=plotter, plot_title="Eta")
    xp.train.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='training')

    xp.val = mlogger.Container()
    xp.val.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="validation")
    xp.val.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='validation')
    xp.max_val = mlogger.metric.Maximum(plotter=plotter, plot_title="Accuracy", plot_legend='best-validation')

    xp.test = mlogger.Container()
    xp.test.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='test')

    if args.visdom:
        plotter.set_win_opts("Gamma", {'ytype': 'log'})
        plotter.set_win_opts("Objective", {'ytype': 'log'})

    if args.log:
        # log at each epoch
        xp.epoch.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.epoch.hook_on_update(lambda: save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name)))

        # log after final evaluation on test set
        xp.test.acc.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.test.acc.hook_on_update(lambda: save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name)))

        # save results and model for best validation performance
        xp.max_val.hook_on_new_max(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

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
