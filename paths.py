import math
import random
import numpy
import math
import copy
import itertools
from tqdm import tqdm
from collections import deque
import heapq
import sys
import csv
import ast
import os
import pickle
from lxml import objectify
from lxml import etree
from graph import LatToMerc, LongToMerc, LatLongToMerc, ValidHighways
import plotly.plotly as py
import plotly
import argparse
import geopy.distance

MinLatitude = 56.18765
MaxLatitude = 56.40198
MinLongitude = 43.72705
MaxLongitude = 44.11196


def Dijkstra(adjacencyList, initial):
    path = {}
    nodes = list(adjacencyList.keys())
    # dict.fromkeys(nodes, sys.maxsize)
    dist = {node: sys.maxsize for node in nodes}
    pbar = tqdm()
    PQ = []
    dist[initial] = 0
    distances = {}
    heapq.heappush(PQ, [0, initial])
    while PQ:
        min_node_dist, min_node_id = heapq.heappop(PQ)
        if min_node_dist == dist[min_node_id]:
            for edge in adjacencyList[min_node_id].keys():
                if dist[min_node_id] + adjacencyList[min_node_id][edge] < dist[edge]:
                    dist[edge] = dist[min_node_id] + \
                        adjacencyList[min_node_id][edge]
                    heapq.heappush(PQ, [dist[edge], edge])
                    path[edge] = min_node_id
                    distances[edge] = adjacencyList[min_node_id][edge]
        pbar.update(1)
    pbar.close()
    return path, distances


def LevitAlgorithm(adjacencyList, initial):
    nodes = list(adjacencyList.keys())
    pbar = tqdm()
    D = dict.fromkeys(nodes, sys.maxsize)
    ID = dict.fromkeys(nodes, 2)
    paths = {}
    D[initial] = 0
    M1 = deque([initial])
    M1f = deque()
    ID[initial] = 1
    while M1 or M1f:
        v = M1f.popleft() if M1f else M1.popleft()
        for edge, dist in adjacencyList[v].items():
            if ID[edge] == 2:
                ID[edge] = 1
                M1.append(edge)
                D[edge] = D[v] + dist
                paths[edge] = v
            elif ID[edge] == 1:
                if (D[edge] > D[v] + dist):
                    D[edge] = D[v] + dist
                    paths[edge] = v
            elif ID[edge] == 0 and D[edge] > D[v] + dist:
                D[edge] = D[v] + dist
                paths[edge] = v
                ID[edge] = 1
                M1f.append(edge)
        ID[v] = 0
        pbar.update(1)
    pbar.close()
    return paths


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def Manhattan(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def Euclid(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def Chebyshev(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return max(abs(x1 - x2), abs(y1 - y2))


def a_star_search(adjacencyList, start, goal, goalID, heuristic):
    length = 0
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goalID:
            break

        for edge, coordinates in adjacencyList[current].items():
            new_cost = cost_so_far[current] + coordinates[0]
            if edge not in cost_so_far or new_cost < cost_so_far[edge]:
                cost_so_far[edge] = new_cost
                priority = new_cost + \
                    heuristic(goal, coordinates[1])  # !!!!!!!!!!!!!!!
                frontier.put(edge, priority)
                came_from[edge] = current

    return came_from, sum(cost_so_far.values())


    
def WriteToPlotlyGraph(osmName, paths, destinations, startNode, is_astar, distances):
    lats = {}
    lons = {}
    iter = 0
    for event, element in etree.iterparse(osmName, tag="node"):
        if event == 'end':
            lats[element.attrib["id"]] = element.attrib['lat']
            lons[element.attrib["id"]] = element.attrib['lon']
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
    roads = []
    iter = 0
    pbar = tqdm()
    for event, element in etree.iterparse(osmName, tag="way"):
        if len(element.xpath("tag[@k = 'highway']")) > 0 and (element.xpath("tag[@k = 'highway']")[0].attrib['v'] in ValidHighways):
            prevRef = None
            iter = 0
            for child in element.iterchildren(tag="nd"):
                if child.attrib['ref'] in lons:
                    if child.getprevious() is not None and prevRef is not None:
                        if iter % 7 == 0 and child.getnext() is not None:
                            roads.append(
                                dict(
                                    type='scattergeo',
                                    lon=[lons[prevRef],
                                        lons[child.attrib["ref"]]],
                                    lat=[lats[prevRef],
                                        lats[child.attrib["ref"]]],
                                    mode='lines',
                                    hoverinfo=element.xpath(
                                        "//way[tag[@k = 'highway']]/tag[@k='name']/@v"),
                                    line=dict(
                                        width=1,
                                        color='rgb(88, 38, 38)',
                                    ),
                                )
                            )
                            prevRef = child.attrib["ref"]
                        iter += 1
                    elif child.getprevious() is None:
                        prevRef = child.attrib["ref"]
                child.clear()
                if child.getnext() is None:
                    while child.getprevious() is not None:
                        del child.getparent()[0]
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
        pbar.update(1)
    pbar.close()
    iter = 0
    pbar = tqdm()
    times = []
    for idx, destination in enumerate(destinations):
        sum = 0
        curNode = '0'
        prevNode = destination
        while curNode != startNode:
            if is_astar:
                if prevNode not in paths[idx] or paths[idx][prevNode] == None:
                    break
                curNode = paths[idx][prevNode]
            else:
                if prevNode not in paths or paths[prevNode] == None:
                    break
                curNode = paths[prevNode]
            sum += distances[prevNode]
            roads.append(
                dict(
                    type='scattergeo',
                    lon=[lons[prevNode], lons[curNode]],
                    lat=[lats[prevNode], lats[curNode]],
                    mode='lines',
                    hoverinfo="text",
                    text=f"Path #{idx}",
                    line=dict(
                        width=1,
                        color='green',
                    ),
                )
            )
            prevNode = curNode
        iter += 1
        pbar.update(1)
        times.append(sum / 40)
        if (iter == 10):
            break
    pbar.close()
    totalSum = 0
    for time in times:
        totalSum += time
    print("Average time:{}".format(totalSum/len(times)))
    layout = dict(
        title='Nizhny Novgorod paths',
        showlegend=False,
        geo=dict(
            scope='world',
            projection=dict(type='Mercur'),
            showland=True,
            lonaxis=dict(
                showgrid=True,
                gridwidth=0.5,
                dtick=0.5,
                range=[43.72705, 44.11196]),
            lataxis=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor="#afb5bf",
                dtick=0.5,
                range=[56.18765, 56.40198]),
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(196, 204, 204)',
        ),
    )
    fig = dict(data=roads, layout=layout)
    plotly.offline.plot(fig, filename='nodes', image='svg',
                        image_filename='nodes.svg')


def WriteToSvgWithPaths(osmName, paths, targetNodes, svgName="graphWithPaths.svg", enlargementKoef=100):
    print("Writing to SVG has started")
    context = etree.iterparse(osmName, events=('end',))
    dwg = svgwrite.Drawing(svgName)

    minCoordY, minCoordX = LatLongToMerc(float(root.getchildren()[0].attrib['minlon']),
                                         float(root.getchildren()[0].attrib['minlat']))
    centerX = ((LatToMerc(float(root.getchildren()[
               0].attrib['maxlat'])) + minCoordX) / 2 - minCoordX) / enlargementKoef
    centerY = (
              (LongToMerc(float(root.getchildren()[0].attrib['maxlon'])) + minCoordY) / 2 - minCoordY) / enlargementKoef
    nodes_dict = {
        node.attrib['id']: (centerX + ((LongToMerc(float(node.attrib['lon'])) - minCoordY) / enlargementKoef - centerY),
                            centerY - ((LatToMerc(float(node.attrib['lat'])) - minCoordX) / enlargementKoef - centerX))
        for node in root.xpath('//node')}
    for way in tqdm(root.xpath("//way[.//tag[@k = 'highway']]")):
        if (way.xpath("tag[@k = 'highway']")[0].attrib['v'] in ValidHighways):
            points = []
            for nd in way.xpath("nd"):
                if (nd.attrib['ref'] in nodes_dict):
                    points.append(nodes_dict[nd.attrib['ref']])
            if len(points) > 0:
                dwg.add(
                    dwg.polyline(points, stroke='brown' if way.xpath("tag[@k = 'highway']")[0].attrib['v'] == 'primary'
                    else 'blue', fill='none', stroke_width=0.8
                    if way.xpath("tag[@k = 'highway']")[0].attrib['v'] == 'primary' else 0.2))
    print("Writing to {0} has ended successfully!".format(svgName))
    dwg.save()
    f.close()


def getNearestPoint(lat, lon, nodesCoordinates):
    min = sys.maxsize
    minCoord = '0'
    for id, node in nodesCoordinates.items():
        if math.sqrt((lat - node[0])**2 + (lon - node[1])**2) < min:
            min = math.sqrt((lat - node[0])**2 + (lon - node[1])**2)
            minCoord = id
    return minCoord


def getCoordinates(point, root):
    tag = root.xpath("//node[@id=\"{0}\"]".format(int(point)))
    return float(tag[0].attrib['lat']), float(tag[0].attrib['lon'])

def getRandomPoints(adjacencyDict):
    keys = adjacencyDict.keys()
    random.shuffle(keys)
    return keys[:100]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "-d", "--dijkstra", help="Check if you want to test Dijkstra algrithm", action='store_true')
    parser.add_argument("-l", "--levit",
                        help="Check if you want to test Levit algorithm",
                        action='store_true')
    parser.add_argument("-t", "--third",
                        help="Check if you want to test Third task",
                        action='store_true')

    parser.add_argument("-ae", "--astare",
                        help="Check if you want to test A* with Euclid",
                        action='store_true')
    parser.add_argument("-am", "--astarm",
                        help="Check if you want to test A* with Manhattan",
                        action='store_true')
    parser.add_argument("-ac", "--astarch",
                        help="Check if you want to test A* with Chebyshev",
                        action='store_true')
    parser.add_argument(
        "-o", "--osmname", help="Name of the OSM file", default="Nizhny_Novgorod.osm")
    parser.add_argument("-sn", "--svgname",
                        help="Name of the SVG file", default="graph.svg")
    parser.add_argument("-e", "--enlargementcoeff",
                        help="Enlargement coefficient", type=int, default=70)
    parser.add_argument(
        "-ln", "--listname", help="Name of the adjacency list file", default="adjacencyList.csv")
    parser.add_argument("-mn", "--matrixname",
                        help="Name of the adjacency matrix file", default="adjacencyMatrix.csv")
    args = parser.parse_args()
    if args.astare or args.astarm or args.astarch:
        a_star = True
    else:
        a_star = False
    print('Loading osm file, please wait...')
    adjacencyDict = {}
    destinations = []
    nodesCoordinates = {}
    if a_star:
        adjacencyListWithWeightsFileName = "adjacencyWeightsWithCoordinates.bin"
    else:
        adjacencyListWithWeightsFileName = "adjacencyWeights.bin"
    destinationsName = "destinations.bin"
    nodesCoordinatesName = "nodesCoordName.bin"
    if not os.path.exists(destinationsName) or not os.path.exists(nodesCoordinatesName) or not os.path.exists(adjacencyListWithWeightsFileName):
        with open("Nizhny_Novgorod.osm", encoding="utf_8_sig") as cityFile:
            xml = cityFile.read()
            root = objectify.fromstring(xml)
            destinations = root.xpath(
                "//node[tag[@v = 'hospital' or contains(@v, 'Больница') or @k = 'healthcare']]/@id")
            with open(destinationsName, 'wb+') as destFile:
                    pickle.dump(destinations, destFile)
            if not os.path.exists(adjacencyListWithWeightsFileName):
                with open("adjacencyList.csv") as adjFile:
                    csv_reader = csv.reader(adjFile)
                    nodesCoordinates = {node.attrib['id']:
                    (float(node.attrib['lat']), float(node.attrib['lon']))
                                        for node in root.xpath("//node")}
                    with open(nodesCoordinatesName, 'wb+') as destFile:
                        pickle.dump(nodesCoordinates, destFile)
                    # skipping headers lines
                    next(csv_reader, None)
                    next(csv_reader, None)
                    for row in tqdm(csv_reader):
                        adjacentNodes = list(ast.literal_eval(row[1]))
                        x, y = nodesCoordinates[row[0]]
                        if not a_star:
                            adjacencyDict[row[0]] = {node: geopy.distance.vincenty(nodesCoordinates[node], (x, y)).km
                                                        for node in adjacentNodes}
                        else:
                            adjacencyDict[row[0]] = {
                            node: [geopy.distance.vincenty(nodesCoordinates[node], (x, y)).km, nodesCoordinates[node]]
                            for node in adjacentNodes}
                with open(adjacencyListWithWeightsFileName, 'wb+') as adjFile:
                    pickle.dump(adjacencyDict, adjFile)
            else:
                with open(adjacencyListWithWeightsFileName, 'rb') as adjFile:
                    adjacencyDict = pickle.load(adjFile)
    else:
        with open(destinationsName, 'rb') as destFile:
            destinations = pickle.load(destFile)
        with open(adjacencyListWithWeightsFileName, 'rb') as adjFile:
            adjacencyDict = pickle.load(adjFile)
        with open(nodesCoordinatesName, 'rb') as destFile:
            nodesCoordinates = pickle.load(destFile)
    userLat = 0.0
    userLon = 0.0
    while userLat < MinLatitude or userLat > MaxLatitude or userLon < MinLongitude or userLon > MaxLongitude: 
        print('Please input your latitude and longitude in following range: \n Latitude: ({},{}) \n Longitude: ({},{}))'.format(MinLatitude, MaxLatitude, MinLongitude, MaxLongitude))
        userLat = float(input("Latitude: "))
        userLon = float(input("Longitude: "))
    cur = getNearestPoint(userLat, userLon, nodesCoordinates)
    print("Chosen point id is {}!".format(cur))


    
    validDestinationsName = "validDestinations.bin"
    #DESTINATIONS LOADER!
    validDestinations = []

    if (os.path.exists(validDestinationsName)):
        with open(validDestinationsName, 'rb') as destFile:
            validDestinations = pickle.load(destFile)
    else:
        destCount = 0
        index = 0
        paths = []
        pbar = tqdm(total=10)
        while destCount < 10:
            if (destinations[index] in adjacencyDict.keys()):
                # paths.append(LevitAlgorithm(adjacencyDict, cur))
                result, cost = a_star_search(adjacencyDict, cur, list(adjacencyDict[destinations[index]].values())[0][1], destinations[index], Euclid)
                if destinations[index] in result:
                    paths.append(result)
                    validDestinations.append(destinations[index])
                    destCount+=1
                    pbar.update(1)
            index +=1
            if index >= len(destinations):
                raise Exception('Vse ploho!')
        pbar.close()
        with open(validDestinationsName, 'wb+') as destFile:
            pickle.dump(validDestinations, destFile)



    paths = []
    import time
    if args.dijkstra:
        print('Starting Dijkstra!')
        start = time.time()
        paths, distances = Dijkstra(adjacencyDict, cur)
        end = time.time()
        print("Elapsed time is {}!".format(end - start))
        print('Starting drawing to Plotly!')
        WriteToPlotlyGraph("Nizhny_Novgorod.osm", paths, validDestinations, cur, args.astare or args.astarm or args.astarch, distances)
    
    if args.levit:
        print('Starting Levit')
        start = time.time()
        paths = LevitAlgorithm(adjacencyDict, cur)
        end = time.time()
        print("Elapsed time is {}!".format(end - start))
        print('Starting drawing to Plotly!')
        WriteToPlotlyGraph("Nizhny_Novgorod.osm", paths, validDestinations, cur, args.astare or args.astarm or args.astarch)
    
    if args.astare:
        print('Starting A* with Euclid')
        paths = []
        start = time.time()
        for destination in validDestinations:
            result, length = a_star_search(adjacencyDict, cur, list(adjacencyDict[destination].values())[0][1], destination, Euclid)
            paths.append(result)
        end = time.time()
        print("Elapsed time is {}!".format(end - start))
        print('Starting drawing to Plotly!')
        WriteToPlotlyGraph("Nizhny_Novgorod.osm", paths, validDestinations, cur, args.astare or args.astarm or args.astarch)

    if args.astarm:
        print('Starting A* with Manhattan')
        paths = []
        start = time.time()
        for destination in validDestinations:
            result, length = a_star_search(adjacencyDict, cur, list(adjacencyDict[destination].values())[0][1], destination, Manhattan)
            paths.append(result)
        end = time.time()
        print("Elapsed time is {}!".format(end - start))
        print('Starting drawing to Plotly!')
        WriteToPlotlyGraph("Nizhny_Novgorod.osm", paths, validDestinations, cur, args.astare or args.astarm or args.astarch) 

    if args.astarch:
        print('Starting A* with Chebyshev')
        paths = []
        start = time.time()
        for destination in validDestinations:
            result, length = a_star_search(adjacencyDict, cur, list(adjacencyDict[destination].values())[0][1], destination, Chebyshev)
            paths.append(result)
        end = time.time()
        print("Elapsed time is {}!".format(end - start))
        print('Starting drawing to Plotly!')
        WriteToPlotlyGraph("Nizhny_Novgorod.osm", paths, validDestinations, cur, args.astare or args.astarm or args.astarch) 
