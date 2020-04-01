<div>
  <img src="figs/logo.png" width="200">
</div>

# More Interactive Weak Supervision with FlyingSquid

FlyingSquid is a new framework for automatically building models from multiple
noisy label sources.
Users write functions that generate noisy labels for data, and FlyingSquid uses
the agreements and disagreements between them to learn a _label model_ of how
accurate the _labeling functions_ are.
The label model can be used directly for downstream applications, or it can be
used to train a powerful end model:

<div>
  <img src="figs/System Diagram.png" width="800">
</div>

FlyingSquid can be used to build models for all sorts of tasks, including text
applications, video analysis, and online learning.
Check out our [blog post](http://hazyresearch.stanford.edu/flyingsquid) and paper on
[arXiv](https://arxiv.org/abs/2002.11955)
for more details!

## Getting Started
* Quickly [install](#installation) FlyingSquid
* Check out the [examples](examples/) folder for tutorials and some simple code
examples

## Sample Usage
```Python
from flyingsquid.label_model import LabelModel
import numpy as np

L_train = np.load('...')

m = L_train.shape[1]
label_model = LabelModel(m)
label_model.fit(L_train)

preds = label_model.predict(L_train)
```

## Installation

We recommend using `conda` to install FlyingSquid:

```
git clone https://github.com/HazyResearch/flyingsquid.git

cd flyingsquid

conda env create -f environment.yml
conda activate flyingsquid

pip install -e .

cd ..
```

Alternatively, you can install the dependencies yourself:
* [Pgmpy](http://pgmpy.org/)
* [PyTorch](https://pytorch.org/) (only necessary for the PyTorch integration) 

And then install the actual package:
```
git clone https://github.com/HazyResearch/flyingsquid.git

cd flyingsquid

pip install -e .

cd ..
```

## Citation

If you use our work or found it useful, please cite our [arXiv paper](https://arxiv.org/abs/2002.11955) for now:
```
@article{fu2020fast,
  author = {Daniel Y. Fu and Mayee F. Chen and Frederic Sala and Sarah M. Hooper and Kayvon Fatahalian and Christopher R\'e},
  title = {Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods},
  journal = {arXiv preprint arXiv:2002.11955},
  year = {2020},
}
```
