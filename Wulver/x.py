#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak
import random
import string
import arachne_development.graph as njit
import arachne_development.methods as methods


def time_ak_diameter():

    cfg = ak.get_config()

    HomeDir="/rhome/zhihui/"
    MtxEdgeList=[ [14496,5242,2,0,HomeDir+"Adata/SNAP/ca-GrQc.mtx"],\
            [68993773,4847571,2,0,HomeDir+"Adata/SNAP/soc-LiveJournal1.mtx"],\
            ]

    start = time.time()
    for i in MtxEdgeList:
        FileName=i[4]
        print(Edges,",",Vertices,",",Columns,",",Directed,",",str(FileName))
        Graph=methods.read_matrix_market_file( str(FileName),False,False)
        diameter = njit.graph_diameter(Graph)
        print("diameter =",diameter)
    return


def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of suffix array building: C= suffix_array(V)")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    return parser


    
if __name__ == "__main__":
    import sys
    
    parser = create_parser()
    args = parser.parse_args()
    ak.verbose = False
    ak.connect(args.hostname, args.port)
    time_ak_diameter()
    ak.shutdown()
