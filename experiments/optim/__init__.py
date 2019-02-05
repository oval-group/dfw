import torch.optim

from dfw import DFW
from dfw.baselines import BPGrad


def get_optimizer(args, parameters):
    """
    Available optimizers:
    - SGD
    - Adam
    - Adagrad
    - AMSGrad
    - DFW
    - BPGrad
    """
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.eta, weight_decay=args.l2,
                                    momentum=args.momentum, nesterov=bool(args.momentum))
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.l2)
    elif args.opt == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=args.eta, weight_decay=args.l2)
    elif args.opt == "amsgrad":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.l2, amsgrad=True)
    elif args.opt == 'dfw':
        optimizer = DFW(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.l2)
    elif args.opt == 'bpgrad':
        optimizer = BPGrad(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.l2)
    else:
        raise ValueError(args.opt)

    print("Optimizer: \t {}".format(args.opt.upper()))

    optimizer.gamma = 1
    optimizer.eta = args.eta

    if args.load_opt:
        state = torch.load(args.load_opt)['optimizer']
        optimizer.load_state_dict(state)
        print('Loaded optimizer from {}'.format(args.load_opt))

    return optimizer


def decay_optimizer(optimizer, decay_factor=0.1):
    if isinstance(optimizer, torch.optim.SGD):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor
        # update state
        optimizer.eta = optimizer.param_groups[0]['lr']
    else:
        raise ValueError
