import csv
from queue import PriorityQueue;
from collections import defaultdict;
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

def astar_time(start, end):
    # Begin your code (Part 6)
    """
    Build a graph with edge.csv, and implement A* time search with priority_queue
    Remind that priority_queue sorts by the first elements of tuple, sum_of_timecost_and_heuristictime is set as the first element
    """
    max_speed = 0
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
            # record the max_speed_limit of all roads
            max_speed = float(e_row[3]) if (float(e_row[3])>max_speed) else max_speed
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
    pqueue.put((float(h_func[start][0][dest])/max_speed, 0, start, [start])) 
    visited = set()
    while not pqueue.empty():
        # tuple in priority_queue stores (goalproximity_of_time, time_from_start_to_current, current_node, current_path_list)
        # Since priority_queue sorts by the first elements of the tuple, heuristic_of_time is set as the first element
        (ht, time, cur, path) = pqueue.get()
        if cur not in visited:
            # mark current node as visited
            visited.add(cur)
            # if reach end_node, return data
            if  cur == end:
                return path, time, len(visited)
            # traverse through neighbors of current_node
            for neighbor, neighbor_distance, neighbor_speed_limit in graph[cur]:
                if neighbor not in visited:
                    # store neighbor_node and its information in priority_queue, traverse its neighbors later
                    # ht is the neighbor's update time + heristic_time of neighbor
                    neighbor_time = (neighbor_distance/neighbor_speed_limit)*3600/1000
                    straight_line = float(h_func[neighbor][0][dest])
                    heuristic_time = straight_line/max_speed
                    pqueue.put((time+neighbor_time+heuristic_time, time+neighbor_time, neighbor, path+[neighbor]))
    return [], 0, 0
    # End your code (Part 6)


if __name__ == '__main__':
    path, time, num_visited = astar_time(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
