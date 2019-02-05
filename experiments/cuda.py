import os
try:
    import waitGPU
    ngpu = int(os.environ['NGPU']) if 'NGPU' in os.environ else 1
    waitGPU.wait(nproc=0, interval=10, ngpu=ngpu, gpu_ids=[2,3])
except ImportError:
    print('Failed to import waitGPU --> no automatic scheduling on GPU')
    pass
import torch  # import torch *after* waitGPU.wait()


def set_cuda(args):
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        torch.zeros(1).cuda()  # for quick initialization of process on device
