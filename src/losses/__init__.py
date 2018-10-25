import torch.nn as nn
from losses.hinge import MultiClassHingeLoss, set_smoothing_enabled


def get_loss(args):
    if args.loss == 'svm':
        loss_fn = MultiClassHingeLoss()
    elif args.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError

    print('L2 regularization: \t {}'.format(args.l2))
    print('\nLoss function:')
    print(loss_fn)

    if args.cuda:
        loss_fn = loss_fn.cuda()

    return loss_fn
