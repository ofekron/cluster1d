from collections import defaultdict

import numpy as np


def default_sortedenumarate(items):
    return np.array(sorted(zip(items, range(len(items)))))

def default_distance(sortedenumarate):
    return np.concatenate([[0], sortedenumarate[1:, 0] - sortedenumarate[:-1, 0]])

class Cluster:
    def __init__(self, items, cluster):
        self.indices = list(map(int, cluster))
        self.items = [items[i] for i in self.indices]
        self.count = len(self.items)
        self.range = (min(self.items), max(self.items))
        self.ranged = self.range[1] - self.range[0]

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()


def to_adj(edges):
    adj=defaultdict(set)
    for u,v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def connected_components(vertices,adj):
    V=set(vertices)
    components=list()
    def component(v):
        visited = set([v])
        toVisit = adj[v]
        while toVisit:
            v=toVisit.pop()
            visited.add(v)
            toVisit|=adj[v]-visited
        return visited
    while V:
        v=V.pop()
        c=component(v)
        components.append(c)
        V-=c
    return components



def heatmap1d(items, levels=3,sortedenumarate=default_sortedenumarate,distances=default_distance):
    """
    Based on cluster1d this function returns a tree with clustering levels s.t maximum distance is taken according to levels count
    For example, given levels=3 the tree levels would be :
    root - cluster full of all items
    clusters with the 2/3th of maximal distance
    clusters with the 1/3th of maximal distance

    :param items:
        the items to cluster
    :param levels:
        how many levels should the tree have - the more levels the higher overhead. more levels gives a more accurate heatmap, default to 3
    :param sortedenumarate:
        either a function that takes items, enumarate them and sort the enumarate by item values - or precomputed such iterable
    :param distances:
        either a function that takes sortedenumarate, and computes the distance between them - or precomputed such iterable

    :return:
        The heat map tree
    """
    ranged=max(items) - min(items)
    n=len(items)
    if callable(sortedenumarate):
        sortedenumarate=sortedenumarate(items)
    if callable(distances):
        distances=distances(sortedenumarate)
    sortedd=list(sorted(np.unique(distances)))
    step=(len(sortedd)-1)/levels
    result=[]
    for i in range(1,levels):
        leveld=sortedd[int(i * step)]
        level=__cluster1d(items, sortedenumarate, distances, leveld)
        result += [(c.range[0], -i,0, c) for c in level]
        result += [(c.range[1], i,1) for c in level]
    maxi=0

    class TreeNode:
        def __init__(self, parent, data):
            self.parent = parent
            self.depth = 0 if parent is None else parent.depth + 1
            self.children = []
            self.data = data
            self.strength = (0 if parent is None else parent.strength) + (data.count / n) / ((data.ranged + 1) / ranged)
        def __repr__(self):
            r='\t' * self.depth + str(self.data.range if type(self.data) is not str else self.data) + " " + str(self.strength) + '\n'
            for c in self.children:
                r+=str(c)
            return r

    root=parent=TreeNode(None,Cluster(items,range(n)))
    for p in sorted(result):
        if p[2]==0:
            curr=TreeNode(parent,p[3])
            curr.parent.children.append(curr)
            parent=curr
            if curr.strength>maxi:
                maxi=curr.strength
        else:
            parent = parent.parent
    root.maxi=maxi
    root.n = n
    return root

def plot_heatmap1d(root,axes,cmap='Reds'):
    """
    helper method to plot the heatmap
    :param root:
        the root returned by heatmap1d
    :param axes:
        the matplotlib axes to use
    :return:
    """
    from matplotlib import pyplot as plt
    from matplotlib import colors
    from matplotlib import cm as cmx
    jet = plt.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=0, vmax=root.maxi)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    q = list([root])
    while q:
        n = q.pop()
        c = n.data
        axes.add_patch(plt.Rectangle((0, c.range[0]), root.n, c.range[1] - c.range[0], alpha=1,
                                          facecolor=scalarMap.to_rgba(n.strength)))
        q += n.children

def __cluster1d(items,sortedenumarate,distances,max_distance):
    a1 = np.concatenate([[0], sortedenumarate[1:, 1]])
    a2 = np.concatenate([[-1], sortedenumarate[:-1, 1]])
    distances = np.array(list(zip(distances, a1, a2)))
    distances = np.array(sorted(distances, key=lambda x: x[0]))
    edges = distances[1:, 1:][distances[1:, 0] <= max_distance]
    adj = to_adj(edges)
    return [Cluster(items, c) for c in connected_components(adj.keys(), adj)]





def cluster1d(items, max_distance,sortedenumarate=default_sortedenumarate,distances=default_distance):
    """
        Clustering algorithm comparable to DBSCAN, using O(n) space and O(nlogn) time complexities.
    :param items:
        the items to cluster
    :param max_distance:
        maximal distance between items in the same cluster
    :param sortedenumarate:
        either a function that takes items, enumarate them and sort the enumarate by item values - or precomputed such iterable
    :param distances:
        either a function that takes sortedenumarate, and computes the distance between them - or precomputed such iterable
    :return:
        Cluster1D object, which is a list of Clusters
    """

    if callable(sortedenumarate):
        sortedenumarate=sortedenumarate(items)
    if callable(distances):
        distances=distances(sortedenumarate)

    return __cluster1d(items,sortedenumarate,distances,max_distance)

if __name__ == '__main__':
    items=[1,2,3,4,5,11,12,3,1,11,12,34,12,123,124,151,17,156,157,12,19,2,21,23,45,16,123,1]
    print(heatmap1d(items,4))