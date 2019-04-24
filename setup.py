from setuptools import setup, find_packages

__version__ = "1.0.a"

setup(name='dfw',
      description='Implementation of the Deep Frank Wolfe (DFW) algorithm',
      author='Leonard Berrada',
      packages=find_packages(),
      license="GNU General Public License",
      url='https://github.com/oval-group/dfw',
      version=str(__version__),
      install_requires=["numpy",
                        "nltk",
                        "torchvision>=0.2",
                        "torch>=1.0",
                        "tqdm",
                        "mlogger",
                        "waitGPU"])
