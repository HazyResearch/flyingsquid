'''
This example code shows how to train a FlyingSquid label model for video data.

It loads some labeling functions to detect Tennis Rallies from the tutorials
folder, and trains up a label model.

You can run this file from the examples folder.
'''

from flyingsquid.label_model import LabelModel
import numpy as np

L_train = np.load('../tutorials/L_train_video.npy')
L_dev = np.load('../tutorials/L_dev_video.npy')
Y_dev = np.load('../tutorials/Y_dev_video.npy')

# Model three frames at a time
v = 3

# Six labeling functions per frame
m_per_frame = 6

# Total number of labeling functions is m_per_frame * v
m = m_per_frame * v

# Figure out how many sequences we're going to have
n_frames_train = L_train.shape[0]
n_frames_dev = L_dev.shape[0]

n_seqs_train = n_frames_train // v
n_seqs_dev = n_frames_dev // v

# Resize and reshape matrices
L_train_seqs = L_train[:n_seqs_train * v].reshape((n_seqs_train, m))
L_dev_seqs = L_dev[:n_seqs_dev * v].reshape((n_seqs_dev, m))
Y_dev_seqs = Y_dev[:n_seqs_dev * v].reshape((n_seqs_dev, v))

# Create the label model with temporal dependencies
label_model = LabelModel(
    m,
    v = v,
    y_edges = [ (i, i + 1) for i in range(v - 1) ],
    lambda_y_edges = [ (i, i // m_per_frame) for i in range(m) ]
)

label_model.fit(L_train_seqs)

probabilistic_labels = label_model.predict_proba_marginalized(L_dev_seqs)
preds = [ 1. if prob > 0.5 else -1. for prob in probabilistic_labels ]
accuracy = np.sum(preds == Y_dev[:n_seqs_dev * v]) / (n_seqs_dev * v)

print('Label model accuracy: {}%'.format(int(100 * accuracy)))
