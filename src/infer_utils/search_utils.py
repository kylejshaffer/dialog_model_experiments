import heapq

import numpy as np

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

class Beam(object):
#For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
#This is so that if two prefixes have equal probabilities then a complete sentence is preferred over an incomplete one since (0.5, False) < (0.5, True)
 
    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width
 
    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)
     
    def __iter__(self):
        return iter(self.heap)

