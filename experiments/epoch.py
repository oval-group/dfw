import torch

from tqdm import tqdm
from dfw.losses import set_smoothing_enabled
from utils import accuracy, regularization


def train(model, loss, optimizer, loader, args, xp):

    model.train()

    for metric in xp.train.metrics():
        metric.reset()

    for x, y in tqdm(loader, disable=not args.tqdm, desc='Train Epoch',
                     leave=False, total=len(loader)):
        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)

        # forward pass
        scores = model(x)

        # compute the loss function, possibly using smoothing
        with set_smoothing_enabled(args.smooth_svm):
            loss_value = loss(scores, y)

        # backward pass
        optimizer.zero_grad()
        loss_value.backward()

        # optimization step
        optimizer.step(lambda: float(loss_value))

        # monitoring
        batch_size = x.size(0)
        xp.train.acc.update(accuracy(scores, y), weighting=batch_size)
        xp.train.loss.update(loss(scores, y), weighting=batch_size)
        xp.train.gamma.update(optimizer.gamma, weighting=batch_size)

    xp.train.eta.update(optimizer.eta)
    w_norm = torch.sqrt(sum(p.norm() ** 2 for p in model.parameters()))
    xp.train.reg.update(0.5 * args.l2 * xp.train.weight_norm.value ** 2)
    xp.train.obj.update(xp.train.reg.value + xp.train.loss.value)
    xp.train.timer.update()

    print('\nEpoch: [{0}] (Train) \t'
          '({timer:.2f}s) \t'
          'Obj {obj:.3f}\t'
          'Loss {loss:.3f}\t'
          'Acc {acc:.2f}%\t'
          .format(int(xp.epoch.value),
                  timer=xp.train.timer.value,
                  acc=xp.train.acc.value,
                  obj=xp.train.obj.value,
                  loss=xp.train.loss.value))

    for metric in xp.train.metrics():
        metric.log(time=xp.epoch.value)


@torch.autograd.no_grad()
def test(model, loader, args, xp):
    model.eval()

    if loader.tag == 'val':
        xp_group = xp.val
    else:
        xp_group = xp.test

    for metric in xp_group.metrics():
        metric.reset()

    for x, y in tqdm(loader, disable=not args.tqdm,
                     desc='{} Epoch'.format(loader.tag.title()),
                     leave=False, total=len(loader)):
        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)
        scores = model(x)
        xp_group.acc.update(accuracy(scores, y), weighting=x.size(0))

    xp_group.timer.update()

    print('Epoch: [{0}] ({tag})\t'
          '({timer:.2f}s) \t'
          'Obj ----\t'
          'Loss ----\t'
          'Acc {acc:.2f}% \t'
          .format(int(xp.epoch.value),
                  tag=loader.tag.title(),
                  timer=xp_group.timer.value,
                  acc=xp_group.acc.value))

    if loader.tag == 'val':
        xp.max_val.update(xp.val.acc.value).log(time=xp.epoch.value)

    for metric in xp_group.metrics():
        metric.log(time=xp.epoch.value)
