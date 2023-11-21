import csv
from collections import deque, defaultdict;
edgeFile = 'edges.csv'

def bfs(start, end):
# Begin your code (Part 1)
    """
    Build a graph with edge.csv, and implement BFS with queue(FIFO)
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

    # initialize visited set and queue
    visited = {start} 
    queue = deque([(start, [start], 0)]) 
    while queue:
        # tuple in queue stores (current_node, current_path_list, distance_from_start_to_current)
        (cur, path, distance) = queue.popleft() 
        # traverse through neighbors of current_node
        for neighbor, neighbor_distance, neighbor_speed_limit in graph[cur]: 
            if neighbor == end:
                return path+[neighbor], distance+neighbor_distance, len(visited)
            if neighbor not in visited:
                # store neighbor_node and its information in queue, traverse its neighbors later
                queue.append((neighbor, path+[neighbor], distance+neighbor_distance))
                visited.add(neighbor)
    return [], 0, 0
# End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
