# Deep Frank-Wolfe For Neural Network Optimization

This repository contains the implementation of the paper [Deep Frank-Wolfe For Neural Network Optimization](https://openreview.net/forum?id=SyVU6s05K7) in pytorch. If you use this work for your research, please cite the paper:

```
@Article{berrada2018deep,
  author       = {Berrada, Leonard and Zisserman, Andrew and Kumar, M Pawan},
  title        = {Deep Frank-Wolfe For Neural Network Optimization},
  journal      = {Under review},
  year         = {2018},
}
```

The DFW algorithm is a first-order optimization algorithm for deep neural networks. To use it for your learning task, consider the two following requirements:
* the loss function has to be convex piecewise linear function (e.g. multi-class SVM [as implemented here](src/losses/hinge.py#L5), or l1 loss)
* the optimizer needs access to the value of the loss function of the current mini-batch [as shown here](src/epoch.py#L31)

Beside these requirements, the optimizer can be used as plug-and-play, and its independent code is available in [src/optim/dfw.py](src/optim/dfw.py)

## Requirements

This code has been tested for pytorch 0.4.1 in python3. Detailed requirements are available in `requirements.txt`.

## Reproducing the Results

* To reproduce the CIFAR experiments: `VISION_DATA=[path/to/your/cifar/data] python scripts/reproduce_cifar.py`
* To reproduce the SNLI experiments: follow the [preparation instructions](https://github.com/lberrada/InferSent/tree/c4ded441cf701c256126c5283e4381abb8271792) and run  `python scripts/reproduce_snli.py`

Note that SGD benefits from a hand-designed learning rate schedule. In contrast, all the other optimizers (including DFW) automatically adapt their steps and rely on the tuning of the initial learning rate only.
On average, you should obtain similar results to the ones reported in the paper (there might be some variance on some instances of CIFAR experiments):

### CIFAR-10:

<table>
<tr><th>Wide Residual Networks </th><th>Densely Connected Networks</th></tr>
<tr><td>

| Optimizer | Test Accuracy (%) |
| --------- | :--------------:  |
| Adagrad   | 86.07             |
| Adam      | 84.86             |
| AMSGrad   | 86.08             |
| BPGrad    | 88.62             |
| **DFW**   | **90.18**         |
| SGD       | 90.08             |

</td><td>

| Optimizer | Test Accuracy (%) |
| --------- | :--------------:  |
| Adagrad   | 87.32             |
| Adam      | 88.44             |
| AMSGrad   | 90.53             |
| **BPGrad**| **90.85**         |
| DFW       | 90.22             |
| **SGD**   | **92.02**         |

</td></tr> </table>

### CIFAR-100:

<table>
<tr><th>Wide Residual Networks </th><th>Densely Connected Networks</th></tr>
<tr><td>

| Optimizer | Test Accuracy (%) |
| --------- | :--------------:  |
| Adagrad   | 57.64             |
| Adam      | 58.46             |
| AMSGrad   | 60.73             |
| BPGrad    | 60.31             |
| **DFW**   | **67.83**         |
| SGD       | 66.78             |

</td><td>

| Optimizer | Test Accuracy (%) |
| --------- | :--------------:  |
| Adagrad   | 56.47             |
| Adam      | 64.61             |
| AMSGrad   | 68.32             |
| BPGrad    | 59.36             |
| **DFW**   | **69.55**         |
| **SGD**   | **70.33**         |

</td></tr> </table>

### SNLI:

<table>
<tr><th>CE Loss</th><th>SVM Loss</th></tr>
<tr><td>

| Optimizer | Test Accuracy (%) |
| --------- | :--------------:  |
| Adagrad   | 83.8              |
| Adam      | 84.5              |
| AMSGrad   | 84.2              |
| BPGrad    | 83.6              |
| DFW       | -                 |
| SGD       | 84.7              |
| SGD*      | 84.5              |

</td><td>

| Optimizer | Test Accuracy (%) |
| --------- | :--------------:  |
| Adagrad   | 84.6              |
| Adam      | 85.0              |
| AMSGrad   | 85.1              |
| BPGrad    | 84.2              |
| **DFW**   | **85.2**          |
| **SGD**   | **85.2**          |
| SGD*      | -                 |

</td></tr> </table>

## Acknowledgments

We use the following third-part implementations:
* [InferSent](https://github.com/facebookresearch/InferSent).
* [DenseNets](https://github.com/andreasveit/densenet-pytorch).
* [Wide ResNets](https://github.com/xternalz/WideResNet-pytorch).
