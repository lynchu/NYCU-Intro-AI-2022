import csv
from queue import PriorityQueue;
from collections import defaultdict;
edgeFile = 'edges.csv'

def ucs(start, end):
# Begin your code (Part 3)
    """
    Build a graph with edge.csv, and implement Uniform Cost Search(USC) with priority_queue
    Just to remind that priority_queue sorts by the first elements of tuple, distance is set as the first element
    """
    with open("edges.csv", 'r') as edgeFile:
        data = csv.reader(edgeFile)
        # skip the first line(header) of the data
        next(data)
        # declare graph as defaultdict(list) to store nodes
        graph = defaultdict(list)
        for row in data:
            # store the node and corresponding information
            # transform data type to int or float, since the original data type is string
            graph[int(row[0])].append((int(row[1]), float(row[2]), float(row[3])))
            # row[0]:start, row[1]:end, row[2]:distance, row[3]:speed limit

    # initialize visited set and priority queue
    pqueue = PriorityQueue()
    pqueue.put((0, start, [start])) 
    visited = set()
    while not pqueue.empty():
        # tuple in priority_queue stores (distance_from_start_to_current, current_node, current_path_list)
        # Since priority_queue sorts by the first elements of the tuple, distance is set as the first element
        (distance, cur, path) = pqueue.get()
        if cur not in visited:
            # mark current node as visited
            visited.add(cur)
            # if reach end_node, return data
            if  cur == end:
                return path, distance, len(visited)
            # traverse through neighbors of current_node
            for neighbor, neighbor_distance, neighbor_speed_limit in graph[cur]:
                if neighbor not in visited:
                    # store neighbor_node and its information in priority_queue, traverse its neighbors later
                    pqueue.put((distance+neighbor_distance, neighbor, path+[neighbor]))
    return [], 0, 0

# End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
