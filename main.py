# CSCI 323
# Assignment 4
# heteng zhang

import sys
import time
from copy import deepcopy
from os import listdir
from os.path import isfile, join


def print_graph(graph, sep=' '):
    str_graph = ''
    for row in range(len(graph)):
        str_graph += sep.join([str(c) for c in graph[row]]) + '\n'
    return str_graph


def read_graph(file_name):
    with open(file_name, 'r') as file:
        graph = []
        lines = file.readlines()

        for line in lines:
            costs = line.split(' ')
            row = []
            for cost in costs:
                row.append(int(cost))
            graph.append(row)
        return graph


def print_graph(graph, sep=' '):
    str_graph = ''

    for row in range(len(graph)):
        str_graph += sep.join([str(c) for c in graph[row]]) + '\n'

    return str_graph


def is_symmetric(graph):
    num_vertices = len(graph)
    for i in range(num_vertices):
        for j in range(num_vertices):
            if graph[i][j] != graph[j][i]:
                return False
    return True


def desc_graph(graph):
    num_vertices = len(graph)
    message = ''
    message += 'Number of vertices = ' + str(num_vertices) + '\n'

    non_zero = 0

    for i in range(num_vertices):
        for j in range(num_vertices):
            if graph[i][j] > 0:
                non_zero += 1

    num_edges = int(non_zero / 2)
    message += 'Number of edges = ' + str(num_edges) + '\n'
    message += 'Symmetric = ' + str(is_symmetric(graph)) + '\n'
    return message


# https://www.geeksforgeeks.org/implementation-of-dfs-using-adjacency-matrix/
def dfs_util(graph, v, visited):
    print(v, end=' ')
    visited[v] = True

    for i in range(len(graph)):
        if graph[v][i] != 0 and not visited[i]:
            dfs_util(graph, i, visited)


def dfs(graph):
    print('DFS: ')
    visited = [False] * len(graph)
    dfs_util(graph, 0, visited)
    print('\n')


# https://www.geeksforgeeks.org/implementation-of-bfs-using-adjacency-matrix/
def bfs_util(graph, start):
    v = len(graph)
    visited = [False] * v
    q = [start]
    visited[start] = True
    while q:
        vis = q[0]
        print(vis, end=' ')
        q.pop(0)
        for i in range(v):
            if (graph[vis][i] != 0 and
                    (not visited[i])):
                q.append(i)
                visited[i] = True


def bfs(graph):
    print('Bfs: ')
    bfs_util(graph, 0)
    print('\n')


# https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
def printMST(graph, parent):
    V = len(graph)
    print("Prims MST")
    print("Edge \tWeight")
    for i in range(1, V):
        print(parent[i], ",", i, "\t", graph[i][parent[i]])


def minKey(graph, key, mstSet):
    V = len(graph)
    min = sys.maxsize

    for v in range(V):
        if key[v] < min and mstSet[v] == False:
            min = key[v]
            min_index = v

    return min_index


def primMST(graph):
    V = len(graph)
    min = sys.maxsize
    key = [min] * V
    parent = [None] * V  # Array to store constructed MST , Make key 0 so that this vertex is picked as first vertex
    key[0] = 0
    mstSet = [False] * V

    parent[0] = -1  # First node is always the root of

    for cout in range(V):
        u = minKey(graph, key, mstSet)
        mstSet[u] = True
        for v in range(V):
            if graph[u][v] > 0 and mstSet[v] == False and key[v] > graph[u][v]:
                key[v] = graph[u][v]
                parent[v] = u

    printMST(graph, parent)


# https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot

    else:
        parent[yroot] = xroot
        rank[xroot] += 1


def KruskalMST(graph, V):
    result = []
    i = 0
    e = 0
    graph = sorted(graph, key=lambda item: item[2])
    parent = []
    rank = []
    for node in range(V):
        parent.append(node)
        rank.append(0)
    while e < V - 1:
        u, v, w = graph[i]
        i = i + 1
        x = find(parent, u)
        y = find(parent, v)
        if x != y:
            e = e + 1
            result.append([u, v, w])
            union(parent, rank, x, y)
    minimumCost = 0
    print("Edges in the constructed MST")
    for u, v, weight in result:
        minimumCost += weight
        print("%d -- %d == %d" % (u, v, weight))
    print("KruskalMST ", minimumCost)


def kruskal_mst(graph):
    print('Kruskal mst: ')
    KruskalMST(graph_to_tuple(graph), len(graph))

def graph_to_tuple(graph):
    list_tuple = []
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] != 0:
                list_tuple.append([i, j, graph[i][j]])
    return list_tuple

def printmatrix(graph):
    for i in range(0, len(graph)):
        for j in range(0, len(graph)):
            print(graph[i][j], end='\t')
        print()


# https://stackoverflow.com/questions/43375515/breadth-first-search-with-adjacency-matrix
def dijkstra(graph):
    num_s = len(graph)
    dist = [[sys.maxsize for _ in range(num_s)] for _ in range(num_s)]
    pred = [[0 for _ in range(num_s)] for _ in range(num_s)]

    for i in range(len(graph)):
        for j in range(len(graph)):
            dist[i][j] = graph[i][j]  # path of length 1, i.e. just the edge
            pred[i][j] = i  # predecessor will be vertex i
            if dist[i][j] == 0:
                dist[i][j] = sys.maxsize
        dist[i][i] = 0  # no cost
        pred[i][i] = 0  # indicates end of path

    for k in range(num_s):
        for i in range(num_s):
            for j in range(num_s):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    pred[i][j] = pred[k][j]
                    print_graph(pred)
                    dist[i][j] = dist[i][k] + dist[k][j]
    print('\n Dijkstra 2d dist array')
    printmatrix(dist)
    print('\nDijkstra pred array for each vertex')
    printmatrix(pred)
    #return [pred, dist]

#  class note
def floyd_apsp(graph):
    num_v = len(graph)
    dist = [[sys.maxsize for _ in range(num_v)] for _ in range(num_v)]
    pred = [[0 for _ in range(num_v)] for _ in range(num_v)]

    for i in range(len(graph)):
        for j in range(len(graph)):
            dist[i][j] = graph[i][j]
            pred[i][j] = i
            if dist[i][j] == 0:
                dist[i][j] = sys.maxsize
        dist[i][i] = 0  # no cost
        pred[i][i] = -1  # indicates end of path

    for k in range(num_v):
        for i in range(num_v):
            for j in range(num_v):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    pred[i][j] = pred[k][j]
                    print_graph(pred)
                    dist[i][j] = dist[i][k] + dist[k][j]
    print('\nFloyd 2d dist array')
    printmatrix(dist)
    print('\nFloyd pred array ')
    printmatrix(pred)


def analyze_graph(file_name):
    graph = read_graph(file_name)
    algos = [dfs, bfs, primMST, kruskal_mst, dijkstra, floyd_apsp]
    output_file_name = file_name[0:-4 + len(file_name)] + '_report.txt'
    sys.stdout = open(output_file_name, 'w')
    print('Analysis of graph: ' + file_name + '\n\n')
    str_graph = print_graph(graph)
    print(str_graph + '\n')
    graph_descrip = desc_graph(graph)
    print(graph_descrip + '\n')

    for algo in algos:
        start_time = time.time()
        algo(deepcopy(graph))
        end_time = time.time()
        net_time = (end_time - start_time) * 1000000
        print('Time: ', float(net_time))

    sys.stdout.close()


def main():
    mypath = "C:\\Users\\heten\\PycharmProjects\\assignment4"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in files:
        if file[0:5] == 'graph' and file.find('_report') < 0:
            analyze_graph(file)


if __name__ == '__main__':
    main()
