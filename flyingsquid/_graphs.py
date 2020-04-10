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
    Functions to check whether we can solve this graph structure.
    '''
    
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