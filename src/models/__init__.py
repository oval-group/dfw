import os
import torch
from .densenet import DenseNet3
from .wide_resnet import WideResNet
from collections import OrderedDict


def get_model(args):
    assert args.dataset in ('cifar10', 'cifar100')

    if args.densenet:
        model = DenseNet3(args.depth, args.n_classes, args.growth, bottleneck=bool(args.bottleneck))
    elif args.wrn:
        model = WideResNet(args.depth, args.n_classes, args.width)
    else:
        raise NotImplementedError

    if args.load_model:
        state = torch.load(args.load_model)['model']
        new_state = OrderedDict()
        for k in state:
            # naming convention for data parallel
            if 'module' in k:
                v = state[k]
                new_state[k.replace('module.', '')] = v
            else:
                new_state[k] = state[k]
        model.load_state_dict(new_state)
        print('Loaded model from {}'.format(args.load_model))

    # Number of model parameters
    args.nparams = sum([p.data.nelement() for p in model.parameters()])
    print('Number of model parameters: {}'.format(args.nparams))

    if args.cuda:
        if args.parallel_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    return model


def load_best_model(model, xp):
    best_model_file = '{}/best_model.pkl'.format(xp.name_and_dir)
    if os.path.exists(best_model_file):
        best_model_state = torch.load(best_model_file)['model']
        model.load_state_dict(best_model_state)
        print('Loaded best model from {}'.format(best_model_file))
    else:
        print('Could not find best model')
