# top-import for cuda device initialization
from cuda import set_cuda

import logger

from cli import parse_command
from losses import get_loss
from utils import get_xp, set_seed
from data import get_data_loaders
from models import get_model, load_best_model
from optim import get_optimizer, decay_optimizer
from epoch import train, test


def main(args):

    set_cuda(args)
    set_seed(args)

    loader_train, loader_val, loader_test = get_data_loaders(args)
    loss = get_loss(args)
    model = get_model(args)
    optimizer = get_optimizer(args, parameters=model.parameters())
    xp = get_xp(args, model, optimizer)

    for i in range(args.epochs):
        xp.Epoch.update(1).log()

        train(model, loss, optimizer, loader_train, xp, args)
        test(model, loader_val, xp, args)

        if (i + 1) in args.T:
            decay_optimizer(optimizer, args.decay_factor)

    load_best_model(model, xp)
    test(model, loader_test, xp, args)


if __name__ == '__main__':
    args = parse_command()
    with logger.stdout_to("{}/log.txt".format(args.xp_name)):
        main(args)
