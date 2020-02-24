from flyingsquid.label_model import LabelModel
import torch
import torch.nn as nn

class FSLoss(nn.Module):
    '''
    Expose FlyingSquid as a loss function.
    The loss function takes sequences: one sequence of outputs of your end model,
    and another sequence of LF votes.
    
    Let `v` be the length of the sequence.
    We will compute BCEWithLogitsLoss, averaged over every element of the
    sequence (and over every sequence in the batch).

    Let `m` be the number of labeling functions.

    Let `batch_size` be the size of your batch during training.

    The loss function will take two arguments: `outputs` and `weak_labels`.
    * The shape of `outputs` will be `batch_size * v`
    * The shape of `weak_labels` will be `batch_size * v * m`

    ```
    # outputs will be batch_size * v
    # weak_labels will be batch_size * v * m
    loss(outputs, weak_labels)
    ```

    The loss function will keep a buffer of N sequences of previous weak labels
    that it's seen.
    Each step, the loss function does the following:
    * For each sequence in the batch (zip over everything in outputs and weak_labels):
      * Add the sequence from `weak_labels` to the buffer (kicking out the oldest
        items in the buffer)
      * Use the triplet method over everything in the buffer (buffer needs to be on
        the CPU) to get probabilistic labels, a tensor of shape `T` (put the tensor
        onto device)
      * For each element in the sequence, compute `BCEWithLogitsLoss` between the
        output and the probabilistic label
      * Return the average over losses in the sequence

    When the dataloader isn't shuffling data, this amounts to "streaming"
    
    Args:
        m: number of LF's
        v: number of Y tasks
        task_deps: edges between the tasks. (i, j) in y_edges means that
            there is an edge between y_i and y_j.
        lf_task_deps: edges between LF's and tasks. (i, j) in lambda_y_edges
            means that there is an edge between lambda_i and y_j.
        lf_deps: edges between LF's. (i, j) in lambda_edges means that
            there is an edge between lambda_i and lambda_j.
        cb: the class balance
        allow_abstentions: if True, all abstentions in LF votes
        device: which device to store the loss/gradients
        buffer_capacity: how many sequences of LF's to cache
        update_frequency: how often to retrain the label model
        clamp_vals: if True, clamp the probabilities out of FlyingSquid to 0.
            or 1.
        triplets: if specified, use this set of triplets for the triplet method
        pos_weight: if specified, set the weight of the positive class to this
            in BCEWithLogitsLoss
        
    Example::
        
        T = ...                # length of a sequence
        m = m_per_task * T     # m_per_task LF's per frame
        
        # this creates a new triplet label model under the hood
        criterion = FSLoss(    
            m, T, 
            [(i, i + 1) for i in range(T - 1)], # chain dependencies for tasks
            [(i + m_per_task * j, j)            # LF's have dependencies to the frames they vote on
               for i in range(m_per_task) for j in range(v)], 
            [],                                 # no dependencies between LF's
            cb = ...                            # pass in class balance if you need to
        )
        
        model = ...            # end model

        frame_sequence = [...] # sequence of T frames
        lf_votes = [...]       # (T, m) vector of LF votes
        
        model_outputs = [      # run the model on each frame
            model(frame)
            for frame in frame_sequence
        ]
        
        # This caches the votes in lf_votes, retrains the label model if necessary, and
        #   generates probabilistic labels for each frame from the LF votes.
        # Then, `BCEWithLogitsLoss` on the model outputs and probabilistic labels is used
        #   to generate the loss value that can be backpropped.
        loss = criterion(      
            torch.tensor([model_outputs]),
            torch.tensor([lf_votes])
        )
        loss.backward()
    '''
    
    def __init__(self, m, v, task_deps, lf_task_deps, lf_deps,
                 Y_dev=None, cb=None, allow_abstentions = False, device='cpu',
                 buffer_capacity=100, update_frequency=10, clamp_vals=False,
                 triplets=None, pos_weight=None):
        super(WSLoss, self).__init__()
        self.m = m
        self.v = v
        self.task_deps = task_deps
        self.lf_task_deps = lf_task_deps
        self.lf_deps = lf_deps
        self.Y_dev = Y_dev
        self.cb = cb
        self.device = device
        self.clamp_vals = clamp_vals
        
        self.tlm = TripletLabelModel(m, v, task_deps, lf_task_deps, lf_deps,
                                     allow_abstentions = allow_abstentions, triplets=triplets)
        
        self.criterion = nn.BCEWithLogitsLoss() if pos_weight is None else nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        self.buffer_capacity = buffer_capacity
        self.update_frequency = update_frequency
        
        # register buffer for LF outputs
        self.register_buffer('lf_buffer', torch.zeros((buffer_capacity, m), dtype=torch.long))
        
        # register buffer to keep track of how many items
        self.register_buffer('buffer_size', torch.zeros(1, dtype=torch.long))
        
        # reigster buffer to keep track of where you are
        self.register_buffer('buffer_index', torch.zeros(1, dtype=torch.long))
        
    def forward(self, predictions, weak_labels, update_frequency = None):
        '''
        Generate probabilistic labels from the weak labels, and use `BCEWithLogitsLoss` to
        get the actual loss value for end model training.
        Also caches the LF votes, and re-trains the label model if necessary (depending on
        update_frequency).
        
        Args:
            predictions: A (batch_size, v)-sized tensor of model outputs. For sequences,
                v is usually the length of the sequence.
            weak_labels: A (batch_size, m)-sized tensor of weak labels.
            
        Returns:
            Computes BCEWithLogitsLoss on every item in the batch (for each item, computes it
            between the v model outputs and the v probabilistic labels), and returns the
            average.
        '''
        update_frequency = update_frequency if update_frequency else self.update_frequency
        
        output = torch.tensor(0., requires_grad=True, device=self.device)
        
        for i, (prediction, label_vector) in enumerate(zip(predictions, weak_labels)):
            self.lf_buffer[self.buffer_index] = label_vector
            if self.buffer_size < self.buffer_capacity:
                self.buffer_size += 1
                
            if (self.buffer_index % update_frequency) == 0:
                L_train = self.lf_buffer.cpu().numpy()[:self.buffer_size]

                self.tlm.fit(
                    L_train,
                    Y_dev = self.Y_dev,
                    class_balance = self.cb
                )
                
            self.buffer_index += 1
            if self.buffer_index == self.buffer_capacity:
                self.buffer_index = torch.tensor(0)
            
            labels = self.tlm.predict_proba_marginalized(
                [label_vector.cpu().numpy()], verbose=False)
            if self.clamp_vals:
                labels[0] = [1. if pred >= 0.5 else 0. for pred in labels[0]]
            
            label_tensor = torch.tensor(labels[0], requires_grad=True, device=self.device).view(prediction.shape)
            
            output = output + self.criterion(
                prediction,
                label_tensor)
        
        return output / predictions.shape[0]


class MajorityVoteLoss(nn.Module):
    '''
    Expose majority vote as a loss function (for baselines).

    Let `m` be the number of labeling functions.

    Let `batch_size` be the size of your batch during training.

    The loss function will take two arguments: `outputs` and `weak_labels`.
    * The shape of `outputs` will be `batch_size`
    * The shape of `weak_labels` will be `batch_size * m`

    ```
    # outputs will be batch_size
    # weak_labels will be batch_size * m
    loss(outputs, weak_labels)
    ```

    The loss function will keep a buffer of N sequences of previous weak labels
    that it's seen.
    Each step, the loss function does the following:
    * For each sequence in the batch (zip over everything in outputs and weak_labels):
      * Add the sequence from `weak_labels` to the buffer (kicking out the oldest
        items in the buffer)
      * Use the triplet method over everything in the buffer (buffer needs to be on
        the CPU) to get probabilistic labels, a tensor of shape `T` (put the tensor
        onto device)
      * For each element in the sequence, compute `BCEWithLogitsLoss` between the
        output and the probabilistic label
      * Return the average over losses in the sequence

    When the dataloader isn't shuffling data, this amounts to "streaming"
    
    Args:
        device: which device to store the loss/gradients
        
    Example::
        
        m = ...                # number of LF's
        
        # this creates a new triplet label model under the hood
        criterion = MajorityVoteLoss(    
            device = ...
        )
        
        model = ...            # end model

        frame_sequence = [...] # sequence of T frames
        lf_votes = [...]       # (T, m) vector of LF votes
        
        model_outputs = [      # run the model on each frame
            model(frame)
            for frame in frame_sequence
        ]
        
        # This caches the votes in lf_votes, retrains the label model if necessary, and
        #   generates probabilistic labels for each frame from the LF votes.
        # Then, `BCEWithLogitsLoss` on the model outputs and probabilistic labels is used
        #   to generate the loss value that can be backpropped.
        loss = criterion(      
            torch.tensor(model_outputs),
            torch.tensor(lf_votes)
        )
        loss.backward()
    '''
    
    def __init__(self, device='cpu', pos_weight=None):
        super(MajorityVoteLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss() if pos_weight is None else nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        self.device = device
        
    def forward(self, predictions, weak_labels, update_frequency = None):
        '''
        Generate probabilistic labels from the weak labels, and use `BCEWithLogitsLoss` to
        get the actual loss value for end model training.
        Also caches the LF votes, and re-trains the label model if necessary (depending on
        update_frequency).
        
        Args:
            predictions: A (batch_size)-sized tensor of model outputs.
            weak_labels: A (batch_size, m)-sized tensor of weak labels.
            
        Returns:
            Computes BCEWithLogitsLoss on every item in the batch (for each item, computes it
            between the v model outputs and the v probabilistic labels), and returns the
            average.
        '''
        
        output = torch.tensor(0., requires_grad=True, device=self.device)
        
        for i, (prediction, label_vector) in enumerate(zip(predictions, weak_labels)):  
            label = (np.sum(label_vector.cpu().numpy()) > 0).astype(float)
            
            label_tensor = torch.tensor(label, requires_grad=True, device=self.device).view(prediction.shape)
            
            output = output + self.criterion(
                prediction,
                label_tensor)
        
        return output / predictions.shape[0]
