import csv
from queue import PriorityQueue;
from collections import defaultdict;
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

def astar(start, end):
# Begin your code (Part 4)
    """
    Build a graph with edge.csv, and implement A* search with priority_queue
    Just to remind that priority_queue sorts by the first elements of tuple, sum_of_pathcost_and_goalproximity is set as the first element
    """
    with open("edges.csv", 'r') as edgeFile, open('heuristic.csv', 'r') as heuristicFile:
        edge_data = csv.reader(edgeFile)
        heuristic_data = csv.reader(heuristicFile)
        # skip the first line(header) of the data
        next(edge_data)
        next(heuristic_data)
        # declare graph and h_func as defaultdict(list) to store nodes and heuristic function of each nodes 
        graph = defaultdict(list)
        h_func = defaultdict(list)
        for e_row in edge_data:
            # store the node and corresponding information
            # transform data type to int or float, since the original data type is string
            # e_row[0]:start, e_row[1]:end, e_row[2]:distance, e_row[3]:speed limit
            graph[int(e_row[0])].append((int(e_row[1]), float(e_row[2]), float(e_row[3])))
        for h_row in heuristic_data:
            # transform data type to int or float, since the original data type is string
            # h_row[0]:node_ID
            # h_row[1]:straight-line distance from node to Big City Shopping Mall(ID: 1079387396)
            # h_row[2]:straight-line distance from node to COSTCO Hsinchu Store(ID: 1737223506)
            # h_row[3]:straight-line distance from node to Nanliao Fighing Port(ID: 8513026827)
            h_func[int(h_row[0])].append((float(h_row[1]), float(h_row[2]), float(h_row[3])))
    
    # determine which column of h_func should be used with the end_node
    if end==1079387396:
        dest = 0
    elif end==1737223506:
        dest = 1
    elif end==8513026827:
        dest = 2

    # initialize visited set and priority queue
    pqueue = PriorityQueue()
    pqueue.put((float(h_func[start][0][dest]), 0, start, [start])) 
    visited = set()
    while not pqueue.empty():
        # tuple in priority_queue stores (sum_of_pathcost_and_goalproximity, distance_from_start_to_current, current_node, current_path_list)
        # Since priority_queue sorts by the first elements of the tuple, heuristic_of_distance is set as the first element
        (hd, distance, cur, path) = pqueue.get()
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
                    # hd is the neighbor's new distance + h_func of neighbor
                    straight_line = float(h_func[neighbor][0][dest])
                    pqueue.put((distance+neighbor_distance+straight_line, distance+neighbor_distance, neighbor, path+[neighbor]))
    return [], 0, 0
# End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
