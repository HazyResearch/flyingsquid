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
    Functions to compute label model parameters from mean parameters.
    '''

    def _generate_e_vector(self, clique):
        '''
        The e vector is a vector of assignments for a particular marginal.
        
        For example, in a marginal with one LF and one Y variable, and no
        abstentions, the e vector entries are:
            [
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1)
            ]
        The first entry of each tuple is the value of the LF, the second
        entry is the value of the Y variagble.
        
        In a marginal with two LFs and one Y variable and no abstentions,
        the entries are:
            [
                (1, 1, 1),
                (1, 1, -1),
                (1, -1, 1),
                (1, -1, -1),
                (-1, 1, 1),
                (-1, 1, -1),
                (-1, -1, 1),
                (-1, -1, -1)
            ]
        
        In a marginal with one Lf, one Y variable, and abstentions:
            [
                (1, 1),
                (0, 1),
                (-1, 1),
                (1, -1),
                (0, -1),
                (-1, -1)
            ]
            
        Two LFs, one Y variable, and abstentions:
            [
                (1, 1, 1),
                (0, 1, 1),
                (-1, 1, 1),
                (1, 0, 1),
                (0, 0, 1),
                (-1, 0, 1),
                (1, -1, 1),
                (0, -1, 1),
                (-1, -1, 1),
                (1, 1, -1),
                (0, 1, -1),
                (-1, 1, -1),
                (1, 0, -1),
                (0, 0, -1),
                (-1, 0, -1),
                (1, -1, -1),
                (0, -1, -1),
                (-1, -1, -1)
            ]
        '''
        lambda_values = [1, 0, -1] if self.allow_abstentions else [1, -1]
        e_vec = [[1], [-1]]
        for i in range(len(clique) - 1):
            new_e_vec = []
            if not self.allow_abstentions:
                for new_val in lambda_values:
                    for e_val in e_vec:
                        new_e_vec.append(e_val + [new_val])
            else:
                for e_val in e_vec:
                    for new_val in lambda_values:
                        new_e_vec.append([new_val] + e_val)
            e_vec = new_e_vec
        e_vec = [ tuple(e_val) for e_val in e_vec ]
        
        return e_vec
    
    def _generate_r_vector(self, clique):
        '''
        The r vector is the vector of probability values that needs to be on the RHS
        of the B_matrix * e_vector = r_vector to make e_vector have the right values.
        
        When there are abstentions, the mapping works as follows:
        * Each probability is some combination of
            P(A * B *  ... * C = 1, D = 0, E = 0, ..., F = 0)
        * The A, B, ..., C can include any LF, and the Y variable.
        * The D, E, ..., F can include any LF
        * Let the A, B, ..., C set be called the "equals one set"
        * Let the D, E, ..., F set be called the "equals zero set"
        * Then, for each entry in the e vector:
          * If there is a -1 in an LF spot, add the LF to the "equals zero set"
          * If there is a 0 in the LF spot, add the LF to the "equals one set"
          * If there is a -1 in the Y variable spot, add it to the "equals one set"
          
        When there are no abstentions, each probability is just defined by the
        "equals one set" (i.e., P(A * B * ... * C = 1)).
        * For each entry in the e vector:
          * If there is a -1 in any spot (LF spot or Y variable), add it to the
            "equals one set"
        '''
        indices = [ int(node.split('_')[1]) for node in clique ]
        lf_indices = sorted(indices[:-1])
        Y_idx = indices[-1]
        Y_val = 'Y_{}'.format(Y_idx)
        
        e_vec = self._generate_e_vector(clique)
        
        r_vec = []
        for e_vec_tup in e_vec:
            # P(a * b * ... * c = 1) for everything in this array
            r_vec_entry_equal_one = []
            # P(a = 0, b = 0, ..., c = 0) for everything in this array
            r_vec_entry_equal_zero = []
            for e_vec_entry, lf_idx in zip(e_vec_tup, lf_indices):
                # if you have abstentions, -1 means add to equal zero, 0 means add to equal one
                if self.allow_abstentions: 
                    if e_vec_entry == -1:
                        r_vec_entry_equal_zero.append('lambda_{}'.format(lf_idx))
                    if e_vec_entry == 0:
                        r_vec_entry_equal_one.append('lambda_{}'.format(lf_idx))
                # otherwise, -1 means add to equal one
                else:
                    if e_vec_entry == -1:
                        r_vec_entry_equal_one.append('lambda_{}'.format(lf_idx))
            if e_vec_tup[-1] == -1:
                r_vec_entry_equal_one.append(Y_val)
        
            entries_equal_one = (
                tuple(['1']) if len(r_vec_entry_equal_one) == 0 else
                tuple(r_vec_entry_equal_one))
            entries_equal_zero = (
                tuple(['0']) if len(r_vec_entry_equal_zero) == 0 else
                tuple(r_vec_entry_equal_zero))
            if self.allow_abstentions:
                r_vec.append((
                    entries_equal_one,
                    entries_equal_zero
                ))
            else:
                if len(r_vec_entry_equal_zero) > 0:
                    print('No abstentions allowed!')
                    exit(1)
                r_vec.append(entries_equal_one)
        
        return r_vec
    
    def _generate_b_matrix(self, clique):
        if not self.allow_abstentions:
            b_matrix_orig = np.array([[1, 1], [1, -1]])
            b_matrix = b_matrix_orig
            for i in range(len(clique) - 1):
                b_matrix = np.kron(b_matrix, b_matrix_orig)
            b_matrix[b_matrix < 0] = 0
            
            return b_matrix
        else:
            a_zero = np.array([
                [1, 1],
                [1, 0]
            ])
            b_zero = np.array([
                [0, 0],
                [0, 1]
            ])
            
            c_matrix = np.array([
                [1, 1, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
            d_matrix = np.array([
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0]
            ])
            
            a_i = a_zero
            b_i = b_zero
            for i in range(len(clique) - 1):
                a_prev = a_i
                b_prev = b_i
                a_i = np.kron(a_prev, c_matrix) + np.kron(b_prev, d_matrix)
                b_i = np.kron(a_prev, d_matrix) + np.kron(b_prev, c_matrix)
                
            return a_i