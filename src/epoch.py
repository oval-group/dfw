import torch

from tqdm import tqdm
from losses import set_smoothing_enabled
from utils import log_metrics, update_metrics, accuracy, regularization


def train(model, loss, optimizer, loader, xp, args):

    model.train()

    xp.Timer_Train.reset()
    stats_dict = {}

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
        stats_dict['loss'] = float(loss(scores, y))
        stats_dict['acc'] = float(accuracy(scores, y))
        stats_dict['gamma'] = float(optimizer.gamma)
        stats_dict['size'] = float(scores.size(0))
        update_metrics(xp, stats_dict)

    xp.Eta.update(optimizer.eta)
    xp.Reg.update(regularization(model, args.l2))
    xp.Obj_Train.update(xp.Reg.value + xp.Loss_Train.value)
    xp.Timer_Train.update()

    print('\nEpoch: [{0}] (Train) \t'
          '({timer:.2f}s) \t'
          'Obj {obj:.3f}\t'
          'Loss {loss:.3f}\t'
          'Acc {acc:.2f}%\t'
          .format(int(xp.Epoch.value),
                  timer=xp.Timer_Train.value,
                  acc=xp.Acc_Train.value,
                  obj=xp.Obj_Train.value,
                  loss=xp.Loss_Train.value))

    log_metrics(xp)


@torch.autograd.no_grad()
def test(model, loader, xp, args):
    model.eval()

    Acc = xp.get_metric(name='acc', tag=loader.tag)
    Timer = xp.get_metric(name='timer', tag=loader.tag)
    Acc.reset()
    Timer.reset()

    for x, y in tqdm(loader, disable=not args.tqdm,
                     desc='{} Epoch'.format(loader.tag.title()),
                     leave=False, total=len(loader)):
        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)
        scores = model(x)
        acc = accuracy(scores, y)
        Acc.update(acc, n=x.size(0))

    Timer.update().log()
    Acc.log()
    print('Epoch: [{0}] ({tag})\t'
          '({timer:.2f}s) \t'
          'Obj ----\t'
          'Loss ----\t'
          'Acc {acc:.2f}% \t'
          .format(int(xp.Epoch.value),
                  tag=loader.tag.title(),
                  timer=Timer.value,
                  acc=Acc.value))

    if loader.tag == 'val':
        xp.Acc_Valbest.update(xp.Acc_Val.value).log()
