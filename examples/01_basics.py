'''
This example code shows a bare-minimum example of how to get FlyingSquid up and
running.

It generates synthetic data from the tutorials folder, and trains up a label
model.

You can run this file from the examples folder.
'''

from flyingsquid.label_model import LabelModel
from tutorials.tutorial_helpers import *

L_train, L_dev, Y_dev = synthetic_data_basics()

m = L_train.shape[1]
label_model = LabelModel(m)

label_model.fit(L_train)

preds = label_model.predict(L_dev).reshape(Y_dev.shape)
accuracy = np.sum(preds == Y_dev) / Y_dev.shape[0]

print('Label model accuracy: {}%'.format(int(100 * accuracy)))
