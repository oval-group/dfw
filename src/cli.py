import os
import argparse


def parse_command():
    parser = argparse.ArgumentParser()

    _add_dataset_parser(parser)
    _add_model_parser(parser)
    _add_optimization_parser(parser)
    _add_loss_parser(parser)
    _add_misc_parser(parser)

    args = parser.parse_args()
    filter_args(args)

    return args


def _add_dataset_parser(parser):
    d_parser = parser.add_argument_group(title='Dataset parameters')
    d_parser.add_argument('--dataset',
                          help='dataset')
    d_parser.add_argument('--train-size', type=int, default=None,
                          help="training data size")
    d_parser.add_argument('--val-size', type=int, default=None,
                          help="val data size")
    d_parser.add_argument('--test-size', type=int, default=None,
                          help="test data size")
    d_parser.add_argument('--no-data-augmentation', dest='data_aug',
                          action='store_false', help='no data augmentation')
    d_parser.set_defaults(data_aug=True)


def _add_model_parser(parser):
    m_parser = parser.add_argument_group(title='Model parameters')
    m_parser.add_argument('--densenet', dest="densenet", action="store_true",
                          help="whether to use densenet on CIFAR")
    m_parser.add_argument('--wrn', dest="wrn", action="store_true",
                          help="whether to use wide residual networks on CIFAR")
    m_parser.add_argument('--depth', type=int, default=None,
                          help="depth of network on densenet / wide resnet")
    m_parser.add_argument('--width', type=int, default=None,
                          help="width of network on wide resnet")
    m_parser.add_argument('--growth', type=int, default=None,
                          help="growth rate of densenet")
    m_parser.add_argument('--bottleneck', dest="bottleneck", action="store_true",
                          help="bottleneck on densenet")
    m_parser.add_argument('--load-model', default=None,
                          help='data file with model')
    m_parser.set_defaults(pretrained=False, wrn=False, densenet=False, bottleneck=True)


def _add_optimization_parser(parser):
    o_parser = parser.add_argument_group(title='Training parameters')
    o_parser.add_argument('--epochs', type=int, default=None,
                          help="number of epochs")
    o_parser.add_argument('--batch-size', type=int, default=None,
                          help="batch size")
    o_parser.add_argument('--eta', type=float, default=0.1,
                          help="initial eta / initial learning rate")
    o_parser.add_argument('--momentum', type=float, default=0.9,
                          help="momentum value for SGD")
    o_parser.add_argument('--opt', type=str, required=True,
                          help="optimizer to use")
    o_parser.add_argument('--T', type=int, default=[-1], nargs='+',
                          help="number of epochs between proximal updates / lr decay")
    o_parser.add_argument('--decay-factor', type=float, default=0.1,
                          help="decay factor for the learning rate / proximal term")
    o_parser.add_argument('--load-opt', default=None,
                          help='data file with opt')


def _add_loss_parser(parser):
    l_parser = parser.add_argument_group(title='Loss parameters')
    l_parser.add_argument('--l2', type=float, default=0,
                          help="l2-regularization")
    l_parser.add_argument('--loss', type=str, default='ce', choices=("svm", "ce"),
                          help="loss function to use ('svm' or 'ce')")
    l_parser.add_argument('--smooth-svm', dest="smooth_svm", action="store_true",
                          help="smooth SVM")
    l_parser.set_defaults(smooth_svm=False)


def _add_misc_parser(parser):
    m_parser = parser.add_argument_group(title='Misc parameters')
    m_parser.add_argument('--seed', type=int, default=None,
                          help="seed for pseudo-randomness")
    m_parser.add_argument('--cuda', type=int, default=1,
                          help="use cuda")
    m_parser.add_argument('--no-visdom', dest='visdom', action='store_false',
                          help='do not use visdom')
    m_parser.add_argument('--server', type=str, default=None,
                          help="server for visdom")
    m_parser.add_argument('--port', type=int, default=9014,
                          help="port for visdom")
    m_parser.add_argument('--xp-name', type=str, default=None,
                          help="name of experiment")
    m_parser.add_argument('--no-log', dest='log', action='store_false',
                          help='do not log results')
    m_parser.add_argument('--debug', dest='debug', action='store_true',
                          help='debug mode')
    m_parser.add_argument('--parallel-gpu', dest='parallel_gpu', action='store_true',
                          help="parallel gpu computation")
    m_parser.add_argument('--no-tqdm', dest='tqdm', action='store_false',
                          help="use of tqdm progress bars")
    m_parser.add_argument('--dump-every-epoch', dest='dump_epoch', action='store_true',
                          help="dump model every epoch")
    m_parser.set_defaults(visdom=True, log=True, debug=False, parallel_gpu=False,
                          tqdm=True, dump_epoch=False)


def filter_args(args):
    args.T = list(args.T)

    if args.debug:
        args.log = args.visdom = False
        args.xp_name = '../debug'
        if not os.path.exists(args.xp_name):
            os.makedirs(args.xp_name)

    if args.log:
        # generate automatic experiment name if not provided
        if args.xp_name is None:
            arch = 'wrn' if args.wrn else 'dn' if args.densenet else 'nn'
            args.xp_name = '../xp/{}-{}-{}'.format(args.dataset, arch, args.opt)
        assert not os.path.exists(args.xp_name), \
            'An experiment already exists at {}'.format(os.path.abspath(args.xp_name))
        os.makedirs(args.xp_name)

    if args.visdom:
        if args.server is None:
            if 'VISDOM_SERVER' in os.environ:
                args.server = os.environ['VISDOM_SERVER']
            else:
                args.visdom = False
                print("Could not find a valid visdom server, de-activating visdom...")

    # default options for densenet
    if args.densenet:
        if not args.depth:
            args.depth = 40
        if not args.growth:
            args.growth = 40
        if not args.batch_size:
            args.batch_size = 64

        if args.epochs is None:
            args.epochs = 300

    # default options for wide residual network
    if args.wrn:
        if not args.depth:
            args.depth = 40
        if not args.width:
            args.width = 4
        if not args.batch_size:
            args.batch_size = 128
        if args.epochs is None:
            args.epochs = 200

    if args.dataset == 'cifar10':
        args.n_classes = 10
    elif args.dataset == 'cifar100':
        args.n_classes = 100
    elif args.dataset == 'snli':
        args.n_classes = 3
