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
    Triplet algorithms as a Mixin. These algorithms recover the mean parameters
    of the graphical model.
    '''
    
    def _triplet_method_single_seed(self, expectations_to_estimate):
        # create triplets for what we need, and return which moments we'll need to compute
        
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
    
    def _triplet_method_mean_median(self, expectations_to_estimate, solve_method):
        exp_to_estimate_list = sorted(list(expectations_to_estimate))
        triplets = []
        
        if self.triplets is None:
            for expectation in exp_to_estimate_list:
                if not self.allow_abstentions:
                    Y_node = expectation[-1]
                else:
                    Y_node = expectation[0][-1]
                
                triplet = [expectation]
                
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
                        if tuple([expectation, second_node, first_node]) in triplets:
                            continue
                        # we found a triplet!
                        triplet = [expectation, first_node, second_node]
                        triplets.append(tuple(triplet))
                        triplet = [expectation, first_node]
                    triplet = [expectation]
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
    
    def _triplet_method_preprocess(self, expectations_to_estimate, solve_method):
        if solve_method == 'triplet':
            return self._triplet_method_single_seed(expectations_to_estimate)
        elif solve_method in [ 'triplet_mean', 'triplet_median' ]:
            return self._triplet_method_mean_median(expectations_to_estimate, solve_method)
    
    def _triplet_method_probabilities(self, triplets, lambda_moment_vals, lambda_zeros,
                                     abstention_probabilities, sign_recovery, solve_method):
        expectation_values = {}
        
        if solve_method == 'triplet':
            pass
        else:
            # each triplet is constructed for the first value in the expectation
            # get all the triplets with the same first value, and take the mean or median
            expectation_value_candidates = {}
            
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

            if solve_method == 'triplet':
                expectation_values[exp1] = (
                    math.sqrt(abs(moment_vals[0] * moment_vals[1] / moment_vals[2])) if moment_vals[2] != 0 else 0)
                expectation_values[exp2] = (
                    math.sqrt(abs(moment_vals[0] * moment_vals[2] / moment_vals[1])) if moment_vals[1] != 0 else 0)
                expectation_values[exp3] = (
                    math.sqrt(abs(moment_vals[1] * moment_vals[2] / moment_vals[0])) if moment_vals[0] != 0 else 0)
            else:
                if exp1 not in expectation_value_candidates:
                    expectation_value_candidates[exp1] = []
                exp_value = (
                    math.sqrt(abs(moment_vals[0] * moment_vals[1] / moment_vals[2])) if moment_vals[2] != 0 else 0)
                expectation_value_candidates[exp1].append(exp_value)
        
        if solve_method in ['triplet_mean', 'triplet_median']:
            for exp in expectation_value_candidates:
                if solve_method == 'triplet_mean':
                    agg_function = np.mean
                if solve_method == 'triplet_median':
                    agg_function = np.median
                expectation_values[exp] = agg_function(expectation_value_candidates[exp])
        
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
                for expectation in sorted(list(expectation_values.keys()))
            }
            
        
        return probabilities, expectation_values