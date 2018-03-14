import argparse
import csv
import inspect
import math

import numpy as np
import svgwrite
from lxml import objectify
from tqdm import tqdm

a = 6378137.0
b = 6356752.3142

f = (a - b) / a
e = math.sqrt(2 * f - f ** 2)
ValidHighways = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
                 "secondary_link", "tertiary", "tertiary_link", "unclassified", "road", "service", "living_street",
                 "residential"]


def LatLongToMerc(lon, lat):
    if lat > 89.5:
        lat = 89.5
    if lat < -89.5:
        lat = -89.5

    rLat = math.radians(lat)
    rLong = math.radians(lon)

    x = a * rLong
    y = a * math.log(
        math.tan(math.pi / 4 + rLat / 2) * ((1 - e * math.sin(rLat)) / (1 + e * math.sin(rLat))) ** (e / 2))
    return x, y


def LatToMerc(lat):
    rLat = math.radians(lat)
    return a * math.log(
        math.tan(math.pi / 4 + rLat / 2) * ((1 - e * math.sin(rLat)) / (1 + e * math.sin(rLat))) ** (e / 2))


def LongToMerc(lon):
    return a * math.radians(lon)


def WriteToSvg(osmName, svgName="graph.svg", enlargementKoef=100):
    print("Writing to SVG has started")
    dwg = svgwrite.Drawing(svgName)
    with open(osmName, encoding="utf_8_sig") as f:
        xml = f.read()
    root = objectify.fromstring(xml)
    minCoordY, minCoordX = LatLongToMerc(float(root.getchildren()[0].attrib['minlon']),
                                         float(root.getchildren()[0].attrib['minlat']))
    nodes_dict = {
        node.attrib['id']: ((LatToMerc(float(node.attrib['lat'])) - minCoordX) / enlargementKoef,
                            (LongToMerc(float(node.attrib['lon'])) - minCoordY) / enlargementKoef)
        for node in root.xpath('//node')}
    for way in tqdm(root.xpath("//way[.//tag[@k = 'highway']]")):
        if (way.xpath("tag[@k = 'highway']")[0].attrib['v'] in ValidHighways):
            points = []
            for nd in way.xpath("nd"):
                if (nd.attrib['ref'] in nodes_dict):
                    points.append(nodes_dict[nd.attrib['ref']])
            if len(points) > 0:
                dwg.add(dwg.polyline(points, stroke='blue', fill='none'))
    print("Writing to {0} has ended successfully!".format(svgName))
    dwg.save()


def WriteAdjacencyListToCSV(csv_file, dict_data):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sep=,'])
        writer.writerow(["Node", "Adjacent nodes"])
        for key, val in tqdm(dict_data.items()):
            if len(val) > 0:
                writer.writerow([key, val])
    print("Writing to {0} has ended successfully!".format(csv_file))


def WriteAdjacencyMatrixToCSV(csv_file, dict_data):
    print("Writing Matrix to CSV has started! (It takes large amount of time and about 3GB of disk space. Be ready! "
          "Also you can break operation at any time using Ctrl + C")
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sep=,'])
        nodes = {node: val for (node, val) in dict_data.items() if len(val) > 0}
        writer.writerow(['', *nodes.keys()])
        nodesLength = len(nodes)
        nodes_list = list(nodes.keys())
        for key, val in tqdm(nodes.items()):
            buf_line = np.zeros(nodesLength, dtype=int)
            for adj in val:
                if (adj in nodes_list):
                    buf_line[nodes_list.index(adj)] = 1
            writer.writerow([key, *buf_line])
    print("Writing to {0} has ended successfully!".format(csv_file))


def WriteToCSV(osmName, adjacencyListName="adjacencyList.csv", adjacencyMatrixName="adjacencyMatrix.csv",
               writeList=True, writeMatrix=False):
    print("Writing to CSV has started")
    with open(osmName, encoding="utf_8_sig") as f:
        xml = f.read()
    root = objectify.fromstring(xml)
    adjacencyDict = {node.attrib['id']: set() for node in root.xpath('//node')}
    for way in tqdm(root.xpath("//way[.//tag[@k = 'highway']]")):
        if (way.xpath("tag[@k = 'highway']")[0].attrib['v'] in ValidHighways):
            way_nodes = way.xpath("nd")
            for idx, nd in enumerate(way_nodes):
                if (nd.attrib["ref"] in adjacencyDict):
                    adjacencyDict[nd.attrib['ref']].update(filter(lambda x: x is not None
                                                                            and x in adjacencyDict,
                                                                  {way_nodes[idx - 1].attrib["ref"]
                                                                   if idx > 0 else None,
                                                                   way_nodes[idx + 1].attrib["ref"]
                                                                   if idx < len(way_nodes) - 1
                                                                   else None}))
    if writeList:
        WriteAdjacencyListToCSV(adjacencyListName, dict_data=adjacencyDict)
    if writeMatrix:
        WriteAdjacencyMatrixToCSV(adjacencyMatrixName, dict_data=adjacencyDict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-s", "--writetosvg", help="Check if you want to write to svg file", action='store_true')
    parser.add_argument("-l", "--writelist",
                        help="Check if you want to write adjacency list to csv",
                        action='store_true')
    parser.add_argument("-m", "--writematrix",
                        help="Check if you want to write adjacency matrix to csv (Use with caution!)",
                        action='store_true')
    parser.add_argument("-o", "--osmname", help="Name of the OSM file", default="Nizhny_Novgorod.osm")
    parser.add_argument("-sn", "--svgname", help="Name of the SVG file", default="graph.svg")
    parser.add_argument("-ln", "--listname", help="Name of the adjacency list file", default="adjacencyList.csv")
    parser.add_argument("-mn", "--matrixname", help="Name of the adjacency matrix file", default="adjacencyMatrix.csv")
    args = parser.parse_args()

    if not (args.writetosvg or args.writelist or args.writematrix):
        print(
            "Please provide at least one argument. Run {0} -h for help".format(inspect.getfile(inspect.currentframe())))

    if args.writetosvg:
        WriteToSvg(args.osmname, svgName=args.svgname)
    if args.writelist or args.writematrix:
        WriteToCSV(args.osmname, args.listname, args.matrixname, args.writelist, args.writematrix)
