import numpy as np
"""
GOAL: Cluster dataset into K different clusters
- unsupervised
- assign points to cluster with nearest mean

STEPS
- Init cluster centers randomly
- Repeat until convergence:
    1. Update Labels: assign points to nearest cluster center 
    2. Update Cluster Centers: set center to the mean of each cluster
"""