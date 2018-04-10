import math
import graph
import sys
import csv
import ast
from lxml import objectify


def Dijkstra(adjacencyList, s):
    n = len(adjacencyList)
    d = [sys.maxsize]*n
    u = [False]*n
    d[s] = 0
    for i in range(n):
        min_weight = sys.maxsize
        minWeightID = -1
        for i in range(n):
            if (d[i] < min_weight and not u[i]):
                min_weight = d[i]
                minWeightID = i
        for adj in adjacencyList[0][minWeightID]:
            d[adj] = min(d[adj], d[minWeightID] + adjacencyList[1][minWeightID])
        u[i] = True
    print(d)
    
def getCoordinates(point, root):
    tag = root.xpath("//node[@id=\"{0}\"]".format(point))
    return float(tag[0].attrib['lat']), float(tag[0].attrib['lon'])
 
if __name__ == "__main__":
    adjacencyDict = {}
    with open("adjacencyList.csv") as adjFile, open("Nizhny_Novgorod.osm", encoding="utf_8_sig") as cityFile:
        xml = cityFile.read()
        root = objectify.fromstring(xml)
        csv_reader = csv.reader(adjFile)
        next(csv_reader, None)
        next(csv_reader, None)
        for row in csv_reader:
            adjacentNodes = map(float, ast.literal_eval(row[1]))
            x, y = getCoordinates(row[0], root)
            adjacencyDict[float(row[0])] = [list(adjacentNodes),
            [math.sqrt((getCoordinates(node, root)[0] - x)**2 + (getCoordinates(node, root)[1] - y)**2) for node in adjacentNodes]]#[row[1], [math.sqrt((getCoordinates(node, root)[0] - x)**2 + math.sqrt((getCoordinates(node, root)[1] - y)**2]]
        print("VSE")