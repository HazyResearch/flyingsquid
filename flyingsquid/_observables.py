from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import JointProbabilityDistribution, DiscreteFactor
from itertools import combinations
from flyingsquid.helpers import *
import numpy as np
import math
from tqdm import tqdm
import sys
import random

class Mixin:
    '''
    Functions to compute observable properties.
    '''

    def _compute_class_balance(self, class_balance=None, Y_dev=None):
        # generate class balance of Ys
        Ys_ordered = [ 'Y_{}'.format(i) for i in range(self.v) ]
        cardinalities = [ 2 for i in range(self.v) ]
        if class_balance is not None:
            class_balance = class_balance / sum(class_balance)
            cb = JointProbabilityDistribution(
                Ys_ordered, cardinalities, class_balance
            )
        elif Y_dev is not None:
            Ys_ordered = [ 'Y_{}'.format(i) for i in range(self.v) ]
            vals = { Y: (-1, 1) for Y in Ys_ordered }
            Y_vecs = sorted([
                [ vec_dict[Y] for Y in Ys_ordered ]
                for vec_dict in dict_product(vals)
            ])
            counts = {
                tuple(Y_vec): 0
                for Y_vec in Y_vecs
            }
            for data_point in Y_dev:
                counts[tuple(data_point)] += 1
            cb = JointProbabilityDistribution(
                Ys_ordered, cardinalities,
                [
                    float(counts[tuple(Y_vec)]) / len(Y_dev)
                    for Y_vec in Y_vecs
                ])
        else:
            num_combinations = 2 ** self.v
            cb = JointProbabilityDistribution(
                Ys_ordered, cardinalities, [
                    1. / num_combinations for i in range(num_combinations)
                ])
            
        return cb
        
    def _compute_Y_marginals(self, Y_marginals):
        for marginal in Y_marginals:
            nodes = [ 'Y_{}'.format(idx) for idx in marginal ]
            Y_marginals[marginal] = self.cb.marginal_distribution(
                nodes,
                inplace=False
            )
        
        return Y_marginals
    
    def _compute_Y_equals_one(self, Y_equals_one):
        # compute from class balance
        for factor in Y_equals_one:
            nodes = [ 'Y_{}'.format(idx) for idx in factor ]

            Y_marginal = self.cb.marginal_distribution(
                nodes,
                inplace=False
            )
            vals = { Y: (-1, 1) for Y in nodes }
            Y_vecs = sorted([
                [ vec_dict[Y] for Y in nodes ]
                for vec_dict in dict_product(vals)
            ])

            # add up the probabilities of all the vectors whose values multiply to +1
            total_prob = 0
            for Y_vec in Y_vecs:
                if np.prod(Y_vec) == 1:
                    vector_prob = Y_marginal.reduce(
                        [
                            (Y_i, Y_val if Y_val == 1 else 0) 
                            for Y_i, Y_val in zip(nodes, Y_vec)
                        ],
                        inplace=False
                    ).values
                    total_prob += vector_prob

            Y_equals_one[factor] = total_prob
            
        return Y_equals_one