from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import JointProbabilityDistribution, DiscreteFactor
from itertools import combinations
from flyingsquid.helpers import *
import numpy as np
import math
from tqdm import tqdm
import sys
import random

class LabelModel:
    
    def __init__(self, m, v, y_edges, lambda_y_edges, lambda_edges, allow_abstentions=False,
                 triplets=None, triplet_seed=0):
        '''Initialize the LabelModel with a graph G.
        
        m: number of LF's
        v: number of Y tasks
        y_edges: edges between the tasks. (i, j) in y_edges means that
            there is an edge between y_i and y_j.
        lambda_y_edges: edges between LF's and tasks. (i, j) in lambda_y_edges
            means that there is an edge between lambda_i and y_j.
        lambda_edges: edges between LF's. (i, j) in lambda_edges means that
            there is an edge between lambda_i and lambda_j.
        allow_abstentions: if True, allow abstentions in L_train.
        triplets: if specified, use these triplets
        triplet_seed: if triplets not specified, randomly shuffle the nodes
            with this seed when generating triplets
        '''
        G = MarkovModel()
        # Add LF nodes
        G.add_nodes_from([
            'lambda_{}'.format(i)
            for i in range(m)
        ])
        G.add_nodes_from([
            'Y_{}'.format(i)
            for i in range(v)
        ])
        
        # Add edges
        G.add_edges_from([
            ('Y_{}'.format(start), 'Y_{}'.format(end))
            for start, end in y_edges
        ])
        G.add_edges_from([
            ('lambda_{}'.format(start), 'Y_{}'.format(end))
            for start, end in lambda_y_edges
        ])
        G.add_edges_from([
            ('lambda_{}'.format(start), 'lambda_{}'.format(end))
            for start, end in lambda_edges
        ])
        
        self.m = m
        self.v = v
        self.G = G
        self.junction_tree = self.G.to_junction_tree()
        
        self.nodes = sorted(list(self.G.nodes))
        self.triplet_seed = triplet_seed
        if triplet_seed is not None:
            random.seed(triplet_seed)
            random.shuffle(self.nodes)
        
        self.separator_sets = set([
            tuple(sorted(list((set(clique1).intersection(set(clique2))))))
            for clique1, clique2 in self.junction_tree.edges
        ])
        
        self.allow_abstentions = allow_abstentions
        self.triplets = triplets
    
    # Make this Picklable
    def save(obj):
        return (obj.__class__, obj.__dict__)

    def load(cls, attributes):
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj
        
    def _is_separator(self, srcSet, dstSet, separatorSet):
        '''Check if separatorSet separates srcSet from dstSet.
        
        Tries to find a path from some node in srcSet to some node in dstSet that doesn't
        pass through separatorSet. If successful, return False. Otherwise, return True.
        '''
        def neighbors(node):
            neighbor_set = set()
            for edge in self.G.edges:
                if edge[0] == node:
                    neighbor_set.add(edge[1])
                if edge[1] == node:
                    neighbor_set.add(edge[0])
            return list(neighbor_set)
        
        visited = set()
        for srcNode in srcSet:
            if srcNode in dstSet:
                return False
            queue = [srcNode]

            curNode = srcNode

            while len(queue) > 0:
                curNode = queue.pop()
                if curNode not in visited:
                    visited.add(curNode)
                else:
                    continue

                for neighbor in neighbors(curNode):
                    if curNode == srcNode:
                        continue
                    if neighbor in dstSet:
                        return False
                    if neighbor in separatorSet:
                        continue
                    if neighbor not in visited:
                        queue.push(neighbor)
        
        return True
                    
        
    def _check(self):
        '''Check to make sure we can solve this.
        
        Checks:
        * For each node or separator set in the junction tree:
            There is either only one Y node in the clique, or the clique is made up entirely of Y nodes, since
            we can only estimate marginals where there is at most one Y, unless the entire marginal is
            made up of Y's)
        * For each node or separator set in the junction tree that contains at least one
          lambda node and exactly one Y node:
            The Y node separates the lambda's from at least two other lambda nodes, that are themselves
            separated by Y. To estimate the marginal mu(lambda_i, ..., lambda_j, Y_k), we need to find
            lambda_a, lambda_b such that lambda_a, lambda_b, and the joint (lambda_i, ..., lambda_j) are
            independent conditioned on Y_k. This amounts to Y_k separating lambda_a, lambda_b, and
            (lambda_i, ..., lambda_j). Note that lambda_i, ..., lambda_j do not have to be separated by Y_k.
        
        Outputs: True if we can solve this, False otherwise.
        '''
        def num_Ys(nodes):
            return len([
                node for node in nodes if 'Y' in node
            ])
        
        def num_lambdas(nodes):
            return len([
                node for node in nodes if 'lambda' in node
            ])
        
        def estimatable_clique(clique):
            y_count = num_Ys(clique)
            lambda_count = num_lambdas(clique)
            
            return y_count <= 1 or lambda_count == 0
        
        for clique in self.junction_tree.nodes:
            if not estimatable_clique(clique):
                return False, "We can't estimate {}!".format(clique)
        
        for separator_set in self.separator_sets:
            if not estimatable_clique(clique):
                return False, "We can't estimate {}!".format(separator_set)
        
        # for each marginal we need to estimate, check if there is a valid triplet
        marginals = sorted(list(self.junction_tree.nodes) + list(self.separator_sets))
        for marginal in marginals:
            y_count = num_Ys(marginal)
            lambda_count = num_lambdas(marginal)
            
            if y_count != 1:
                continue
            
            separator_y = [node for node in marginal if 'Y' in node]
            lambdas = [node for node in marginal if 'lambda' in node]
            
            found = False
            for first_node in self.nodes:
                if 'Y' in first_node or first_node in lambdas:
                    continue
                for second_node in self.nodes:
                    if 'Y' in first_node or first_node in lambdas:
                        continue
                        
                    if (self._is_separator(lambdas, [first_node], separator_y) and
                        self._is_separator(lambdas, [second_node], separator_y) and
                        self._is_separator([first_node], [second_node], separator_y)):
                        found = True
                        break
                if found:
                    break
            
            if not found:
                print('Could not find triplet for {}!'.format(marginal))
                return False
        
        return True
    
    def enumerate_ys(self):
        # order to output probabilities
        vals = { Y: (-1, 1) for Y in range(self.v) }
        Y_vecs = sorted([
            [ vec_dict[Y] for Y in range(self.v) ]
            for vec_dict in dict_product(vals)
        ])
        
        return Y_vecs
    
    def _triplet_method_preprocess(self, expectations_to_estimate):
        # create triplets for what we need, and return the moments we'll need
        
        exp_to_estimate_list = sorted(list(expectations_to_estimate))
        if self.triplet_seed is not None:
            random.shuffle(exp_to_estimate_list)
        
        if self.triplets is None:
            expectations_in_triplets = set()
            triplets = []
            for expectation in exp_to_estimate_list:
                # if we're already computing it, don't need to add to a new triplet
                if expectation in expectations_in_triplets:
                    continue

                if not self.allow_abstentions:
                    Y_node = expectation[-1]
                else:
                    Y_node = expectation[0][-1]

                def check_triplet(triplet):
                    return (self._is_separator(triplet[0][:-1], triplet[1][:-1], Y_node) and
                        self._is_separator(triplet[0][:-1], triplet[2][:-1], Y_node) and
                        self._is_separator(triplet[1][:-1], triplet[2][:-1], Y_node))

                triplet = [expectation]
                found = False

                # first try looking at the other expectations that we need to estimate
                for first_node in exp_to_estimate_list:
                    if self.allow_abstentions:
                        # need to check if conditionals are the same
                        if (first_node in triplet or # skip if it's already in the triplet
                            first_node[0][-1] != Y_node or # skip if the Y values aren't the same
                            first_node[1] != expectation[1] or # skip if conditions are different
                            (len(first_node[0]) > 2 and len(expectation[0]) > 2) or # at most one item in the triplet can have length > 2
                            first_node in expectations_in_triplets or # we're already computing this
                            not self._is_separator(expectation[0][:-1], first_node[0][:-1], Y_node)): # not separated
                            continue
                    else:
                        if (first_node in triplet or # skip if it's already in the triplet
                            first_node[-1] != Y_node or # skip if the Y values aren't the same
                            (len(first_node) > 2 and len(expectation) > 2) or # at most one item in the triplet can have length > 2
                            first_node in expectations_in_triplets or # we're already computing this
                            not self._is_separator(expectation[:-1], first_node[:-1], Y_node)): # not separated
                            continue    
                    triplet = [expectation, first_node]
                    # first try looking at the other expectations that we need to estimate
                    for second_node in exp_to_estimate_list:
                        if self.allow_abstentions:
                            if (second_node in triplet or # skip if it's already in the triplet
                                second_node[0][-1] != Y_node or # skip if the Y values aren't the same
                                second_node[1] != expectation[1] or # skip if conditions are different
                                (len(second_node[0]) > 2 and
                                     any(len(exp[0]) > 2 for exp in triplet)) or # at most one item in the triplet can have length > 2
                                second_node in expectations_in_triplets or # we're already computing this
                                not all(self._is_separator(exp[0][:-1], second_node[0][:-1], Y_node) for exp in triplet)): # not separated
                                continue
                        else:
                            if (second_node in triplet or # skip if it's already in the triplet
                                second_node[-1] != Y_node or # skip if the Y values aren't the same
                                (len(second_node) > 2 and
                                     any(len(exp) > 2 for exp in triplet)) or # at most one item in the triplet can have length > 2
                                second_node in expectations_in_triplets or # we're already computing this
                                not all(self._is_separator(exp[:-1], second_node[:-1], Y_node) for exp in triplet)): # not separated
                                continue

                        # we found a triplet!
                        triplet = [expectation, first_node, second_node]
                        found = True
                        break
                    if found:
                        break

                    # otherwise, try everything
                    for second_node in [
                        ((node, Y_node), expectation[1]) if self.allow_abstentions else (node, Y_node)
                        for node in self.nodes
                    ]:
                        if self.allow_abstentions:
                            if (second_node in triplet or # skip if it's already in the triplet
                                second_node[1] != expectation[1] or # skip if conditions are different
                                not all(self._is_separator(exp[0][:-1], second_node[0][:-1], Y_node) for exp in triplet)): # not separated
                                continue
                        else:
                            if (second_node in triplet or # skip if it's already in the triplet
                                not all(self._is_separator(exp[:-1], second_node[:-1], Y_node) for exp in triplet)): # not separated
                                continue

                        # we found a triplet!
                        triplet = [expectation, first_node, second_node]
                        found = True
                        break

                if not found:
                    # try everything
                    for first_node in [
                        ((node, Y_node), expectation[1]) if self.allow_abstentions else (node, Y_node)
                        for node in self.nodes if 'Y' not in node
                    ]:
                        if self.allow_abstentions:
                            if (first_node in triplet or # skip if it's already in the triplet
                                first_node[0][0] in expectation[1] or # skip if the node is part of the condition
                                not self._is_separator(expectation[0][:-1], first_node[0][:-1], Y_node)): # not separated
                                continue
                        else:
                            if (first_node in triplet or # skip if it's already in the triplet
                                not self._is_separator(expectation[:-1], first_node[:-1], Y_node)): # not separated
                                continue 

                        triplet = [expectation, first_node]

                        for second_node in [
                            ((node, Y_node), expectation[1]) if self.allow_abstentions else (node, Y_node)
                            for node in self.nodes if 'Y' not in node
                        ]:
                            if self.allow_abstentions:
                                if (second_node in triplet or # skip if it's already in the triplet
                                    second_node[0][0] in expectation[1] or # skip if the node is part of the condition
                                    not all(self._is_separator(exp[0][:-1], second_node[0][:-1], Y_node) for exp in triplet)): # not separated
                                    continue
                            else:
                                if (second_node in triplet or # skip if it's already in the triplet
                                    not all(self._is_separator(exp[:-1], second_node[:-1], Y_node) for exp in triplet)): # not separated
                                    continue
                            # we found a triplet!
                            triplet = [expectation, first_node, second_node]
                            found = True
                            break

                if found:
                    triplets.append(triplet)
                    for expectation in triplet:
                        expectations_in_triplets.add(expectation)
        else:
            triplets = self.triplets
        
        all_moments = set()
        abstention_probabilities = {}
        
        for exp1, exp2, exp3 in triplets:
            if self.allow_abstentions:
                condition = exp1[1]
                
                moments = [
                    tuple(sorted(exp1[0][:-1] + exp2[0][:-1])),
                    tuple(sorted(exp1[0][:-1] + exp3[0][:-1])),
                    tuple(sorted(exp2[0][:-1] + exp3[0][:-1]))
                ]
                
                indices1 = tuple(sorted([ int(node.split('_')[1]) for node in exp1[0][:-1] ]))
                indices2 = tuple(sorted([ int(node.split('_')[1]) for node in exp2[0][:-1] ]))
                indices3 = tuple(sorted([ int(node.split('_')[1]) for node in exp3[0][:-1] ]))
                
                if indices1 not in abstention_probabilities:
                    abstention_probabilities[indices1] = 0
                if indices2 not in abstention_probabilities:
                    abstention_probabilities[indices2] = 0
                if indices3 not in abstention_probabilities:
                    abstention_probabilities[indices3] = 0
            else:
                # first, figure out which moments we need to compute
                moments = [
                    tuple(sorted(exp1[:-1] + exp2[:-1])),
                    tuple(sorted(exp1[:-1] + exp3[:-1])),
                    tuple(sorted(exp2[:-1] + exp3[:-1]))
                ]
            for moment in moments:
                indices = tuple(sorted([ int(node.split('_')[1]) for node in moment ]))
                
                if indices not in all_moments:
                    all_moments.add(indices)
        
        return triplets, all_moments, abstention_probabilities
    
    def _triplet_method_probabilities(self, triplets, lambda_moment_vals, lambda_zeros,
                                     abstention_probabilities, sign_recovery):
        expectation_values = {}
        for exp1, exp2, exp3 in triplets:
            if self.allow_abstentions:
                moments = [
                    tuple(sorted(exp1[0][:-1] + exp2[0][:-1])),
                    tuple(sorted(exp1[0][:-1] + exp3[0][:-1])),
                    tuple(sorted(exp2[0][:-1] + exp3[0][:-1]))
                ]
            else:
                # first, figure out which moments we need to compute
                moments = [
                    tuple(sorted(exp1[:-1] + exp2[:-1])),
                    tuple(sorted(exp1[:-1] + exp3[:-1])),
                    tuple(sorted(exp2[:-1] + exp3[:-1]))
                ]
            
            moment_vals = [
                lambda_moment_vals[
                    tuple(sorted([ int(node.split('_')[1]) for node in moment ]))
                ]
                for moment in moments
            ]
            
            expectation_values[exp1] = (
                math.sqrt(abs(moment_vals[0] * moment_vals[1] / moment_vals[2])) if moment_vals[2] > 0 else 0)
            expectation_values[exp2] = (
                math.sqrt(abs(moment_vals[0] * moment_vals[2] / moment_vals[1])) if moment_vals[1] > 0 else 0)
            expectation_values[exp3] = (
                math.sqrt(abs(moment_vals[1] * moment_vals[2] / moment_vals[0])) if moment_vals[0] > 0 else 0)
        
        if sign_recovery == 'all_positive':
            # all signs are already positive
            pass
        else:
            print('{} sign recovery not implemented'.format(sign_recovery))
            return
        
        if self.allow_abstentions:
            # probability is 0.5 * (1 + expectation - P(lambda part of factor is zero)) * P(conditional)
            # P(conditional) is 1 if there is no conditional
            probabilities = {}
            for expectation in sorted(list(expectation_values.keys())):
                exp_value = expectation_values[expectation]
                if expectation[1][0] == '0':
                    condition_prob = 1
                else:
                    zero_condition = tuple(sorted([ int(node.split('_')[1]) for node in expectation[1] ]))
                    condition_prob = lambda_zeros[zero_condition]
                
                lambda_factor = tuple(sorted([ int(node.split('_')[1]) for node in expectation[0][:-1] ]))
                abstention_prob = abstention_probabilities[lambda_factor]
                
                probabilities[expectation] = 0.5 * (1 + exp_value - abstention_prob) * condition_prob
        else:
            probabilities = {
                expectation: 0.5 * (1 + expectation_values[expectation])
                for expectation in sorted(list(expectation_value.keys()))
            }
            
        
        return probabilities
    
    def _generator_e_vector(self, clique):
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
        indices = [ int(node.split('_')[1]) for node in clique ]
        lf_indices = sorted(indices[:-1])
        Y_idx = indices[-1]
        Y_val = 'Y_{}'.format(Y_idx)
        
        e_vec = self._generator_e_vector(clique)
        
        r_vec = []
        for e_vec_tup in e_vec:
            # P(a * b * ... * c = 1) for everything in this array
            r_vec_entry_equal_one = []
            # P(a = 0, b = 0, ..., c = 0) for everything in this array
            r_vec_entry_equal_zero = []
            for e_vec_entry, lf_idx in zip(e_vec_tup, lf_indices):
                # if there's a -1 in the slot, add
                if e_vec_entry == -1:
                    r_vec_entry_equal_zero.append('lambda_{}'.format(lf_idx))
                if e_vec_entry == 0:
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
    
    def _lambda_pass(self, L_train, lambda_marginals, lambda_moment_vals, lambda_equals_one,
                    lambda_zeros, abstention_probabilities, verbose = False):
        '''
        Make the pass over L_train.
        
        In this pass, we need to:
        * Compute all the joint marginal distributions over multiple lambda's (lambda_marginals)
        * Compute the probabilities that some set of lambda's are all equal to zero (lambda_zeros)
        * Compute all the lambda moments, including conditional moments (lambda_moment_vals)
        * Compute the probability that the product of some lambdas is zero (abstention_probabilities)
        '''
        
        # do the fast cases first
        easy_marginals = {
            marginal: None
            for marginal in lambda_marginals
            if len(marginal) == 1
        }
        easy_moments = {
            moment: None
            for moment in lambda_moment_vals
            if type(moment[0]) != type(()) and len(moment) <= 2
        }
        easy_equals_one = {
            factor: None
            for factor in lambda_equals_one
            if type(factor[0]) != type(()) and len(factor) == 1
        }
        easy_zeros = {
            condition: None
            for condition in lambda_zeros if len(condition) == 1
        }
        easy_abstention_probs = {
            factor: None
            for factor in abstention_probabilities if len(factor) == 1
        }
        
        means = np.einsum('ij->j', L_train)/L_train.shape[0]
        covariance = np.einsum('ij,ik->jk', L_train, L_train)/L_train.shape[0]
        
        lf_cardinality = 3 if self.allow_abstentions else 2
        lf_values = (-1, 0, 1) if self.allow_abstentions else (-1, 1)
        for marginal in easy_marginals:
            idx = marginal[0]
            counts = [ np.sum(L_train[:,idx] == val) / L_train.shape[0] for val in lf_values ]
            easy_marginal[marginal] = JointProbabilityDistribution(
                [ 'lambda_{}'.format(idx) ], [ lf_cardinality ], counts
            )
            
            if marginal in easy_equals_one:
                easy_equals_one[marginal] = counts[-1]
            if marginal in easy_zeros:
                easy_zeros[marginal] = counts[1]
            if marginal in easy_abstention_probs:
                easy_abstention_probs[marginal] = counts[1]
        for moment in easy_moments:
            if len(moment) == 1:
                easy_moments[moment] = means[moment[0]]
            else:
                easy_moments[moment] = covariance[moment[0]][moment[1]]
        for factor in easy_equals_one:
            if easy_equals_one[factor] is None:
                easy_equals_one[factor] = np.sum(L_train[:,factor[0]] == 1) / L_train.shape[0]
        for condition in easy_zeros:
            if easy_zeros[condition] is None:
                idx = condition[0]
                easy_zeros[condition] = np.sum(L_train[:,idx] == 0) / L_train.shape[0]
        for factor in easy_abstention_probs:
            if easy_abstention_probs[factor] is None:
                idx = factor[0]
                easy_abstention_probs[factor] = np.sum(L_train[:,idx] == 0) / L_train.shape[0]
                
        # time for the remaining cases
        lambda_marginals = {
            key: lambda_marginals[key]
            for key in lambda_marginals
            if key not in easy_marginals
        }
        lambda_moment_vals = {
            key: lambda_moment_vals[key]
            for key in lambda_moment_vals
            if key not in easy_moments
        }
        lambda_equals_one = {
            key: lambda_equals_one[key]
            for key in lambda_equals_one
            if key not in easy_equals_one
        }
        lambda_zeros = {
            key: lambda_zeros[key]
            for key in lambda_zeros
            if key not in easy_zeros
        }
        abstention_probabilities = {
            key: abstention_probabilities[key]
            for key in abstention_probabilities
            if key not in easy_abstention_probs
        }
        
        # for the rest, loop through L_train
        if (len(lambda_marginals) > 0 or len(lambda_moment_vals) > 0 or
            len(lambda_equals_one) > 0 or len(lambda_zeros) > 0 or
            len(abstention_probabilities) > 0):
        
            # figure out which lambda states we need to keep track of
            lambda_marginal_counts = {}
            lambda_marginal_vecs = {}
            lf_values = (-1, 0, 1) if self.allow_abstentions else (-1, 1)
            for lambda_marginal in lambda_marginals:
                nodes = [ 'lambda_{}'.format(idx) for idx in lambda_marginal ]
                vals = { lf: lf_values for lf in nodes }
                lf_vecs = sorted([
                    [ vec_dict[lf] for lf in nodes ]
                    for vec_dict in dict_product(vals)
                ])
                counts = {
                    tuple(lf_vec): 0
                    for lf_vec in lf_vecs
                }
                lambda_marginal_vecs[lambda_marginal] = lf_vecs
                lambda_marginal_counts[lambda_marginal] = counts

            lambda_moment_counts = { moment: 0 for moment in lambda_moment_vals }
            lambda_moment_basis = { moment: 0 for moment in lambda_moment_vals }
            lambda_equals_one_counts = { factor: 0 for factor in lambda_equals_one }
            lambda_equals_one_basis = { factor: 0 for factor in lambda_equals_one }
            lambda_zero_counts = { condition: 0 for condition in lambda_zeros }
            abstention_probability_counts = { factor: 0 for factor in abstention_probabilities }
            
            for data_point in tqdm(L_train) if verbose else L_train:
                for marginal in lambda_marginals:
                    mask = [ data_point[idx] for idx in marginal ]
                    lambda_marginal_counts[marginal][tuple(mask)] += 1
                for moment in lambda_moment_vals:
                    if type(moment[0]) == type(()):
                        pos_mask = [ data_point[idx] for idx in moment[0] ]
                        zero_mask = [ data_point[idx] for idx in moment[1] ]

                        if np.count_nonzero(zero_mask) == 0:
                            lambda_moment_basis[moment] += 1
                        lambda_moment_counts[moment] += np.prod(pos_mask)
                    else:
                        mask = [ data_point[idx] for idx in moment ]
                        lambda_moment_counts[moment] += np.prod(mask)
                        lambda_moment_basis[moment] += 1
                for factor in lambda_equals_one:
                    if type(factor[0]) == type(()):
                        pos_mask = [ data_point[idx] for idx in factor[0] ]
                        zero_mask = [ data_point[idx] for idx in factor[1] ]

                        if np.count_nonzero(zero_mask) == 0:
                            lambda_equals_one_basis[factor] += 1
                            if np.prod(pos_mask) == 1:
                                lambda_equals_one_counts[factor] += 1
                    else:
                        mask = [ data_point[idx] for idx in factor ]
                        if np.prod(mask) == 1:
                            lambda_equals_one_counts[factor] += 1
                        lambda_equals_one_basis[factor] += 1
                for zero_condition in lambda_zeros:
                    zero_mask = [ data_point[idx] for idx in zero_condition ]
                    if np.count_nonzero(zero_mask) == 0:
                        lambda_zero_counts[zero_condition] += 1
                for factor in abstention_probability_counts:
                    zero_mask = [ data_point[idx] for idx in factor ]
                    if np.prod(zero_mask) == 0:
                        abstention_probability_counts[factor] += 1
                        
            lf_cardinality = 3 if self.allow_abstentions else 2
            for marginal in lambda_marginals:
                nodes = [ 'lambda_{}'.format(idx) for idx in marginal ]
                lf_vec = lambda_marginal_vecs[marginal]
                counts = lambda_marginal_counts[marginal]

                lambda_marginals[marginal] = JointProbabilityDistribution(
                    nodes, [ lf_cardinality for node in nodes ],
                    [
                        float(counts[tuple(lf_vec)]) / len(L_train)
                        for lf_vec in lf_vecs
                    ]
                )

            for moment in lambda_moment_vals:
                if lambda_moment_basis[moment] == 0:
                    moment_val = 0
                else:
                    moment_val = lambda_moment_counts[moment] / lambda_moment_basis[moment]
                lambda_moment_vals[moment] = moment_val
                
            for factor in lambda_equals_one:
                if lambda_equals_one_basis[factor] == 0:
                    prob = 0
                else:
                    prob = lambda_equals_one_counts[factor] / lambda_equals_one_basis[factor]
                lambda_equals_one[factor] = prob

            for zero_condition in lambda_zeros:
                lambda_zeros[zero_condition] = lambda_zero_counts[zero_condition] / len(L_train)

            for factor in abstention_probabilities:
                abstention_probabilities[factor] = abstention_probability_counts[factor] / len(L_train)
                
        # update with the easy values
        lambda_marginals.update(easy_marginals)
        lambda_moment_vals.update(easy_moments)
        lambda_equals_one.update(easy_equals_one)
        lambda_zeros.update(easy_zeros)
        abstention_probabilities.update(easy_abstention_probs)
            
        return lambda_marginals, lambda_moment_vals, lambda_equals_one, lambda_zeros, abstention_probabilities
    
    def fit(self, L_train, class_balance=None, Y_dev=None, flip_negative=True, clamp=True, 
            solve_method='triplet',
            sign_recovery='all_positive',
            verbose = False):
        '''Compute the marginal probabilities of each clique and separator set in the junction tree.
        
        L_train: an m x n matrix of LF outputs. L_train[k][i] is the value of \lambda_i on item k.
            1 means positive, -1 means negative. No abstains.
        class_balance: a 2^v vector of the probabilities of each combination of Y values. Sorted in
          lexicographical order (entry zero is for Y_0 = -1, ..., Y_{v-1} = -1, entry one is for
          Y_0 = -1, ..., Y_{v-1} = 1, last entry is for Y_0 = 1, ..., Y_{v-1} = 1).
        Y_dev: a v x |Y_dev| matrix of ground truth examples. If class_balance is not specified, this
          is used to find out the class balance. Otherwise not used.
          If this is not specified, and class_balance is not specified, then class balance is uniform.
          1 means positive, -1 means negative.
        flip_negative: if True, slip sign of negative probabilities
        clamp: if True and flip_negative is not True, set negative probabilities to 0
        solve_method: one of ['triplet', 'independencies']
          If triplet, use the method below and the independencies we write down there.
          If independencies, use the following facts:
            * For any lambda_i: lambda_i * Y and Y are independent for any i, so
              E[lambda_i Y] = E[lambda_i] / E[Y]
            * For any lambda_i, lambda_j: E[lambda_i * lambda_j * Y] = E[lambda_i * lambda_j] * E[Y]
            * For an odd number of lambda's, the first property holds; for an even number, the second
              property holds
          Only triplet implemented right now.
        sign_recovery: one of ['all_positive', 'fully_independent']
          If all_positive, assume that all accuracies that we compute are positive.
          If fully_independent, assume that the accuracy of lambda_0 on Y_0 is positive, and that for
            any lambda_i and lambda_{i+1}, sign(lambda_i lambda_{i+1}) = sign(M_{i,i+1}) where M_{i, i+1}
            is the second moment between lambda_0 and lambda_i.
          If solve_method is independencies, we don't need to do this.
          Only all_positive implemented right now.
        verbose: if True, print out messages to stderr as we make progress
        
        How we go about solving these probabilities (for Triplet method):
          * We assume that we have the joint distribution/class balance of our Y's (or can infer it
            from the dev set).
          * We observe agreements and disagreements between LF's, so we can compute values like
            P(\lambda_i \lambda_j = 1). (For now, we assume that LF's are binary and do not abstain).
          * The only thing we need to estimate now are correlations between LF's and (unseen) Y's -
            values like P(\lambda_i Y_j = 1).
          * Luckily, we have P(\lambda_i Y_j = 1) = 1/2(1 + E[\lambda_i Y_j]). We refer to E[\lambda_i Y_j]
            as the accuracy of \lambda_i on Y_j.
          * And because of the format of our exponential model, we have:
              E[\lambda_i Y_j]E[\lambda_k Y_j] = E[\lambda_i Y_j \lambda_k Y_j] = E[\lambda_i \lambda_k]
            For any \lambda_i, \lambda_k that are conditionally independent given Y_j. This translates to
              Y_j being a separator of \lambda_i and \lambda_k in our graphical model.
            And we can observe E[\lambda_i \lambda_k] (the second moment) from L_train!
          * The algorithm proceeds to estimate the marginal probabilities by picking out triplets of
            conditionally-independent subsets of LF's, and estimating the accuracies of LF's on Y's.
          * Then, to recover the joint probabilities, we can solve a linear system B e = r (written out in latex):
          
              $$\begin{align*}
                \begin{bmatrix}
                1 & 1 & 1 & 1 \\
                1 & 0 & 1 & 0 \\
                1 & 1 & 0 & 0 \\
                1 & 0 & 0 &1
                \end{bmatrix}
                \begin{bmatrix}
                p_{\lambda_i, Y_j}(+1, +1)\\ 
                p_{\lambda_i, Y_j}(-1, +1)  \\ 
                p_{\lambda_i, Y_j}(+1, -1) \\ 
                p_{\lambda_i, Y_j}(-1, -1) \end{bmatrix} = 
                \begin{bmatrix} 1 \\ 
                P(\lambda_{i} = 1) \\ 
                P(Y_j = 1)  \\ 
                \rho_{i, j} \end{bmatrix} .
                \end{align*}$$
            
              The values on the left of the equality are an invertible matrix, and values like
              P(\lambda_i = 1, Y_j = 1), P(\lambda_i = -1, Y_j = 1), etc for the full marginal probability.
              The values on the left of the equality are [1, P(\lambda_i = 1), P(Y_j = 1), P(\lambda_i = Y_j)]^T.
              We can observe or solve for all the values on the right, to solve for the values in the marginal
              probability!
              This can also be extended to multiple dimensions.
            
            Outputs: None.
        '''
        # if abstentions not allowed, check for zero's
        if not self.allow_abstentions:
            if np.count_nonzero(L_train) < L_train.shape[0] * L_train.shape[1]:
                print('Abstentions not allowed!')
                return
            
        # Y marginals to compute
        Y_marginals = {}
        
        # lambda marginals to compute
        lambda_marginals = {}
            
        # marginals will eventually be returned here
        marginals = [
            (clique, None)
            for clique in sorted(list(self.junction_tree.nodes)) + sorted(list(self.separator_sets))
        ]
        
        def num_Ys(nodes):
            if nodes == tuple([1]) or nodes == tuple([0]):
                return 0
            return len([
                node for node in nodes if 'Y' in node
            ])
        
        def num_lambdas(nodes):
            if nodes == tuple([1]) or nodes == tuple([0]):
                return 0
            return len([
                node for node in nodes if 'lambda' in node
            ])
        
        observable_cliques = []
        non_observable_cliques = []
        
        for i, (clique, _) in enumerate(marginals):
            if num_Ys(clique) == 0 or num_lambdas(clique) == 0:
                observable_cliques.append(i)
            else:
                non_observable_cliques.append(i)
        
        # write down everything we need for the observable cliques
        for idx in observable_cliques:
            clique = marginals[idx][0]
            indices = tuple(sorted([ int(node.split('_')[1]) for node in clique ]))
            
            if 'Y' in clique[0]:
                if indices not in Y_marginals:
                    Y_marginals[indices] = None
            else:
                if indices not in lambda_moment_vals:
                    lambda_marginals[indices] = None
                    
        if verbose:
            print('Marginals written down', file=sys.stderr)
                    
        # for each marginal we need to estimate, write down the r vector that we need
        r_vecs = {} # mapping from clique index to the r vector
        r_vals = {} # mapping from a value name (like Y_1 or tuple(lambda_1, Y_1)) to its value
        for idx in non_observable_cliques:
            clique = list(reversed(sorted(marginals[idx][0])))
            r_vec = self._generate_r_vector(clique)
            r_vecs[idx] = r_vec
            for r_val in r_vec:
                if r_val not in r_vals:
                    r_vals[r_val] = None
                    
        if verbose:
            print('R vector written down', file=sys.stderr)
        
        # write down all the sets of zero conditions
        lambda_zeros = {}
        
        # write down the moment values that we need to keep track of when we walk through the L matrix
        Y_equals_one = {}
        lambda_equals_one = {}
        
        # write down which expectations we need to solve using the triplet method
        expectations_to_estimate = set()
        for r_val in r_vals:
            if not self.allow_abstentions or r_val[1] == tuple(['0']):
                equals_one_tup = r_val if not self.allow_abstentions else r_val[0]
                
                if equals_one_tup[0] == '1':
                    # If the value is 1, the probability is just 1
                    r_vals[r_val] = 1
                elif num_Ys(equals_one_tup) != 0 and num_lambdas(equals_one_tup) != 0:
                    # If this contains lambdas and Y's, we can't observe it
                    expectations_to_estimate.add(r_val)
                elif num_Ys(equals_one_tup) != 0:
                    # We need to cache this moment
                    indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_one_tup ]))
                    if indices not in Y_equals_one:
                        Y_equals_one[indices] = None
                elif num_lambdas(equals_one_tup) != 0:
                    # If it contains just lambdas, go through L_train
                    indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_one_tup ]))
                    if indices not in lambda_equals_one:
                        lambda_equals_one[indices] = None
            else:
                # we allow abstentions, and there are clauses that are equal to zero
                equals_one_tup = r_val[0]
                equals_zero_tup = r_val[1]
                if num_lambdas(equals_one_tup) > 0 and num_Ys(equals_one_tup) > 0:
                    # we can't observe this
                    expectations_to_estimate.add(r_val)
                elif num_lambdas(equals_one_tup) > 0:
                    # compute probably some lambda's multiply to one, subject to some zeros
                    pos_indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_one_tup ]))
                    zero_indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_zero_tup ]))
                    
                    tup = (pos_indices, zero_indices)
                    if tup not in lambda_equals_one:
                        lambda_equals_one[tup] = None
                    if zero_indices not in lambda_zeros:
                        lambda_zeros[zero_indices] = None
                else:
                    # compute a Y equals one probability, and multiply it by probability of zeros
                    if equals_one_tup[0] != '1':
                        pos_indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_one_tup ]))
                        if pos_indices not in Y_equals_one:
                            Y_equals_one[pos_indices] = None
                    zero_indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_zero_tup ]))
                    if zero_indices not in lambda_zeros:
                        lambda_zeros[zero_indices] = None
        
        if verbose:
            print('Expectations to estimate written down', file=sys.stderr)
        
        if solve_method == 'triplet':
            triplets, new_moment_vals, abstention_probabilities = self._triplet_method_preprocess(
                expectations_to_estimate)
            self.triplets = triplets
        elif solve_method == 'independencies':
            print('Independencies not implemented yet!')
            return
        
        if verbose:
            print('Triplets constructed', file=sys.stderr)
        
        lambda_moment_vals = {}
        for moment in new_moment_vals:
            if moment not in lambda_moment_vals:
                lambda_moment_vals[moment] = None
        
        # now time to compute all the Y marginals
        self.cb = self._compute_class_balance(class_balance, Y_dev)
        Y_marginals = self._compute_Y_marginals(Y_marginals)
        
        if verbose:
            print('Y marginals computed', file=sys.stderr)
            
        Y_equals_one = self._compute_Y_equals_one(Y_equals_one)
        
        if verbose:
            print('Y equals one computed', file=sys.stderr)
                
        self.Y_marginals = Y_marginals
        self.Y_equals_one = Y_equals_one
        
        # now time to compute the lambda moments, marginals, zero conditions, and abstention probs
        lambda_marginals, lambda_moment_vals, lambda_equals_one, lambda_zeros, abstention_probabilities = self._lambda_pass(
            L_train, lambda_marginals, lambda_moment_vals, lambda_equals_one,
            lambda_zeros, abstention_probabilities, verbose = verbose)
        
        if verbose:
            print('lambda marginals, moments, conditions computed', file=sys.stderr)
            
        self.lambda_marginals = lambda_marginals
        self.lambda_moment_vals = lambda_moment_vals
        self.lambda_equals_one = lambda_equals_one
        self.lambda_zeros = lambda_zeros
        self.abstention_probabilities = abstention_probabilities
            
        # put observable cliques in the right place
        for idx in observable_cliques:
            clique = marginals[idx][0]
            indices = tuple(sorted([ int(node.split('_')[1]) for node in clique ]))
            
            if 'Y' in clique[0]:
                marginal = Y_marginals[indices]
            else:
                marginal = lambda_marginals[indices]
                
            marginals[idx] = (clique, marginal)
        
        # get unobserved probabilities
        if solve_method == 'triplet':
            probability_values = self._triplet_method_probabilities(
                triplets, lambda_moment_vals, lambda_zeros,
                abstention_probabilities, sign_recovery)
        elif solve_method == 'independencies':
            print('Independencies not implemented yet!')
            return
        
        self.probability_values = probability_values
        
        if verbose:
            print('Unobserved probabilities computed', file=sys.stderr)
        
        # put values into the R vectors
        for r_val in r_vals:
            if not self.allow_abstentions or r_val[1] == tuple(['0']):
                equals_one_tup = r_val if not self.allow_abstentions else r_val[0]
                
                if equals_one_tup[0] == '1':
                    # If the value is 1, the probability is just 1
                    pass
                elif num_Ys(equals_one_tup) != 0 and num_lambdas(equals_one_tup) != 0:
                    # If this contains lambdas and Y's, we can't observe it
                    r_vals[r_val] = probability_values[r_val]
                elif num_Ys(equals_one_tup) != 0:
                    # We need to cache this moment
                    indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_one_tup ]))
                    r_vals[r_val] = Y_equals_one[indices]
                elif num_lambdas(equals_one_tup) != 0:
                    indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_one_tup ]))
                    r_vals[r_val] = lambda_equals_one[indices]
            else:
                # we allow abstentions, and there are clauses that are equal to zero
                equals_one_tup = r_val[0]
                equals_zero_tup = r_val[1]
                if num_lambdas(equals_one_tup) > 0 and num_Ys(equals_one_tup) > 0:
                    # we can't observe this
                    r_vals[r_val] = probability_values[r_val]
                elif num_lambdas(equals_one_tup) > 0:
                    # compute lambda moment, subject to some zeros
                    pos_indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_one_tup ]))
                    zero_indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_zero_tup ]))
                    
                    tup = (pos_indices, zero_indices)
                    r_vals[r_val] = lambda_equals_one[tup]
                else:
                    # compute a Y moment, and multiply it by probability of zeros
                    if equals_one_tup[0] != '1':
                        pos_indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_one_tup ]))
                        
                        pos_prob = Y_equals_one[pos_indices]
                    else:
                        pos_prob = 1.
                    zero_indices = tuple(sorted([ int(node.split('_')[1]) for node in equals_zero_tup ]))
                    zero_probs = lambda_zeros[zero_indices]
                    
                    r_vals[r_val] = pos_prob * zero_probs
                    
        self.r_vals = r_vals
                        
        if verbose:
            print('R values computed', file=sys.stderr)
        
        # solve for marginal values
        for idx in non_observable_cliques:
            clique = list(reversed(sorted(marginals[idx][0])))
            r_vec = r_vecs[idx]
            
            r_vec_vals = np.array([ r_vals[exp] for exp in r_vec ])
            
            # e_vec is the vector of marginal values
            e_vec = self._generator_e_vector(clique)
            
            b_matrix = self._generate_b_matrix(clique)
            
            e_vec_vals = np.linalg.inv(b_matrix) @ r_vec_vals
            
            e_vec_val_index = { tup: i for i, tup in enumerate(e_vec) }
            marginal_vals = np.array([
                e_vec_vals[e_vec_val_index[tup]]
                for tup in sorted(e_vec)
            ])
            
            if flip_negative:
                marginal_vals[marginal_vals < 0] = marginal_vals[marginal_vals < 0] * -1
                marginal_vals /= sum(marginal_vals)
            elif clamp:
                marginal_vals[marginal_vals < 0] = 1e-8
                marginal_vals /= sum(marginal_vals)
            
            indices = [ int(node.split('_')[1]) for node in clique ]
            lf_indices = sorted(indices[:-1])
            Y_idx = indices[-1]
            
            variables = [ 'lambda_{}'.format(i) for i in lf_indices ] + [ 'Y_{}'.format(Y_idx) ]
            
            # cardinality 3 for lambda variables if you allow abstentions, 2 for Y's
            cardinalities = [
                3 if self.allow_abstentions else 2
                for i in range(len(lf_indices))
            ] + [2]
            
            marginal = DiscreteFactor(variables, cardinalities, marginal_vals).normalize(inplace = False)
            
            marginals[idx] = (clique, marginal)
        
        self.clique_marginals = marginals[:len(self.junction_tree.nodes)]
        self.separator_marginals = marginals[len(self.junction_tree.nodes):]
        separator_degrees = {
            sep: 0
            for sep in self.separator_sets
        }
        for clique1, clique2 in self.junction_tree.edges:
            separator_degrees[tuple(sorted(list((set(clique1).intersection(set(clique2))))))] += 1
        self.separator_degrees = separator_degrees
        
    def reduce_marginal(self, marginal, data_point):
        lf_vals = [-1, 0, 1] if self.allow_abstentions else [-1, 1]
        params = [
            (var, lf_vals.index(data_point[int(var.split('_')[1])]))
            for var in marginal.variables if 'lambda' in var
        ]
        return marginal.reduce(params, inplace=False) if len(params) > 0 else marginal
    
    def predict_proba(self, L_matrix, verbose=True):
        '''Predict the probabilities of the Y's given the outputs of the LF's.
        
        L_matrix: a m x |Y| matrix of of LF outputs. L_matrix[k][i] is the value of \lambda_i on item k.
            1 means positive, -1 means negative. No abstains.
        
        Let C be the set of all cliques in the graphical model, and S the set of all separator sets.
        Let d(s) for s \in S be the number of maximal cliques that s separates.
        
        Then, we have the following formula for the joint probability:
        
          P(\lambda_1, ..., \lambda_m, Y_1, ..., Y_v) =
              \prod_{c \in C} \mu_c(c) / \prod_{s \in S} [\mu_s(s)]^(d(s) - 1)
        
        Where \mu_c and \mu_s are the marginal probabilities of a clique c or a separator s, respectively.
        We solved for these marginals during the fit function, so now we use them for inference!
        
        Outputs: a 2^v x |Y| matrix of probabilities. The probabilities for the combinations are
          sorted lexicographically.
        '''
        def num_lambdas(nodes):
            return len([
                node for node in nodes if 'lambda' in node
            ])
        
        Y_vecs = self.enumerate_ys()
        numerator_vals_by_lambda_count = []
        max_lambda_count = max([ num_lambdas(clique) for clique, marginal in self.clique_marginals ])
        
        # Compute all marginals that have lambda_count lambdas
        for lambda_count in range(1, max_lambda_count + 1):
            correct_lambda_cliques = [
                (clique, marginal)
                for clique, marginal in self.clique_marginals if num_lambdas(clique) == lambda_count
            ]
            lambda_vals = {
                i: (-1, 0, 1) if self.allow_abstentions else (-1, 1)
                for i in range(lambda_count)
            }
            lambda_vecs = sorted([
                [ vec_dict[i] for i in range(lambda_count) ]
                for vec_dict in dict_product(lambda_vals)
            ])

            # index by Y_vec, clique, and lambda value
            A_lambda = np.zeros((len(Y_vecs), len(correct_lambda_cliques), len(lambda_vecs)))

            for i, Y_vec in enumerate(Y_vecs):
                for j, (clique, marginal) in enumerate(correct_lambda_cliques):
                    lambda_marginal = marginal.reduce(
                        [
                            ('Y_{}'.format(Y_idx), y_val if y_val == 1 else 0)
                            for Y_idx, y_val in enumerate(Y_vec)
                            if 'Y_{}'.format(Y_idx) in clique
                        ],
                        inplace = False
                    )
                    for k, lambda_vec in enumerate(lambda_vecs):
                        A_lambda[i, j, k] = lambda_marginal.reduce(
                            [
                                (clique_node, lambda_val + 1)
                                for clique_node, lambda_val in zip(clique, lambda_vec)
                            ], 
                            inplace=False).values

            indexes = np.array([
                [
                    np.sum([
                        (((data_point[int(node.split('_')[1])]) + 1) * 
                         ((3 if self.allow_abstentions else 2) ** (lambda_count - i - 1)))
                        for i, node in enumerate(clique[:-1])
                    ])
                    for clique, marginal in correct_lambda_cliques
                ]
                for data_point in L_matrix
            ])

            clique_values = A_lambda[:, np.arange(indexes.shape[1]), indexes]

            numerator_values = np.prod(clique_values, axis=2)
            numerator_vals_by_lambda_count.append(numerator_values)
        
        # Compute all marginals that have zero lambdas
        zero_lambda_cliques = [
            (clique, marginal)
            for clique, marginal in self.clique_marginals if num_lambdas(clique) == 0
        ]
        if len(zero_lambda_cliques) > 0:
            A_y = np.zeros((len(Y_vecs), len(zero_lambda_cliques)))
            for i, Y_vec in enumerate(Y_vecs):
                for j, (clique, marginal) in enumerate(zero_lambda_cliques):
                    Y_marginal = marginal.reduce(
                        [
                            ('Y_{}'.format(Y_idx), y_val if y_val == 1 else 0)
                            for Y_idx, y_val in enumerate(Y_vec)
                            if 'Y_{}'.format(Y_idx) in clique
                        ],
                        inplace = False
                    )
                    A_y[i, j] = Y_marginal.values

            y_probs = np.prod(A_y, axis=1)

            numerator_ys = np.array([y_probs,] * L_matrix.shape[0]).T
        
        # Compute all separator marginals
        zero_lambda_separators = [
            (clique, marginal)
            for clique, marginal in self.separator_marginals if num_lambdas(clique) == 0
        ]

        A_y_sep = np.zeros((len(Y_vecs), len(zero_lambda_separators)))
        for i, Y_vec in enumerate(Y_vecs):
            for j, (clique, marginal) in enumerate(zero_lambda_separators):
                Y_marginal = marginal.reduce(
                    [
                        ('Y_{}'.format(Y_idx), y_val if y_val == 1 else 0)
                        for Y_idx, y_val in enumerate(Y_vec)
                        if 'Y_{}'.format(Y_idx) in clique
                    ],
                    inplace = False
                )
                A_y_sep[i, j] = Y_marginal.values ** (self.separator_degrees[clique] - 1)

        y_probs_sep = np.prod(A_y_sep, axis=1)

        denominator_ys = np.array([y_probs_sep,] * L_matrix.shape[0]).T
        
        predictions = numerator_vals_by_lambda_count[0]
        for lambda_numerator in numerator_vals_by_lambda_count[1:]:
            predictions = predictions * lambda_numerator
        if len(zero_lambda_cliques) > 0:
            predictions = predictions * numerator_ys
        predictions = predictions / denominator_ys
        
        normalized_preds = predictions.T / np.array(([predictions.sum(axis = 0),] * len(Y_vecs))).T
        
        return normalized_preds
    
    def predict(self, L_matrix, verbose=True):
        '''Predict the value of the Y's that best fits the outputs of the LF's.
        
        L_matrix: a m x |Y| matrix of LF outputs. L_matrix[k][i] is the value of \lambda_i on item k.
            1 means positive, -1 means negative. No abstains.
        
        Let C be the set of all cliques in the graphical model, and S the set of all separator sets.
        Let d(s) for s \in S be the number of maximal cliques that s separates.
        
        Then, we have the following formula for the joint probability:
        
          P(\lambda_1, ..., \lambda_m, Y_1, ..., Y_v) =
              \prod_{c \in C} \mu_c(c) / \prod_{s \in S} [\mu_s(s)]^(d(s) - 1)
        
        Where \mu_c and \mu_s are the marginal probabilities of a clique c or a separator s, respectively.
        We solved for these marginals during the fit function, so now we use them for inference!
        
        Outputs: a v x |Y| matrix of predicted outputs.
        '''
        
        Y_vecs = self.enumerate_ys()
        combination_probs = self.predict_proba(L_matrix, verbose=verbose)
        most_likely = np.argmax(combination_probs, axis=1)
        preds = np.array(Y_vecs)[most_likely]
        
        return preds
    
    def predict_proba_marginalized(self, L_matrix, verbose=False):
        '''Predict the probabilities of the Y's given the outputs of the LF's, marginalizing out all the
        Y values every time (return a separate probability for +1/-1 for each Y).
        
        L_matrix: a m x |Y| matrix of of LF outputs. L_matrix[k][i] is the value of \lambda_i on item k.
            1 means positive, -1 means negative. No abstains.
        
        Let C be the set of all cliques in the graphical model, and S the set of all separator sets.
        Let d(s) for s \in S be the number of maximal cliques that s separates.
        
        Then, we have the following formula for the joint probability:
        
          P(\lambda_1, ..., \lambda_m, Y_1, ..., Y_v) =
              \prod_{c \in C} \mu_c(c) / \prod_{s \in S} [\mu_s(s)]^(d(s) - 1)
        
        Where \mu_c and \mu_s are the marginal probabilities of a clique c or a separator s, respectively.
        We solved for these marginals during the fit function, so now we use them for inference!
        
        Outputs: a v x |Y| matrix of marginalized probabilities (one probability for each task, for each
          data point). 
        '''
        combination_probs = self.predict_proba(L_matrix, verbose=verbose)
        # construct indices for each task
        Y_vecs = self.enumerate_ys()
        task_indices = [
            [ idx for idx, y_vec in enumerate(Y_vecs) if y_vec[i] == 1 ]
            for i in range(self.v)
        ]
        
        return np.sum(combination_probs[:, task_indices], axis=2).reshape(len(combination_probs) * self.v)
