# FlyingSquid: More Interactive Weak Supervision

<div>
  <img src="figs/System Diagram.png" width="400">
</div>

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
Check out our blog post and paper on arXiv for more details.

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

conda env create -f environment.yml
conda activate flyingsquid

pip install -e .
```

Alternatively, you can install the dependencies yourself:
* [Pgmpy](http://pgmpy.org/)
* [PyTorch](https://pytorch.org/) (only necessary for the PyTorch integration) 

And then install the actual package:
```
git clone https://github.com/HazyResearch/flyingsquid.git

pip install -e .
```
