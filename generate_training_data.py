import ast
import random
import itertools
import numpy as np
from pynauty import *
from copy import deepcopy
from functools import reduce
from collections import OrderedDict

g3 = Graph(number_of_vertices=3, directed=False,
        adjacency_dict = {
            0 : [1,2],
            1 : [0,2],
            2 : [0,1],
            },
        )

g4 = Graph(number_of_vertices=4, directed=False,
        adjacency_dict = {
            0 : [1,2,3],
            1 : [0,2,3],
            2 : [0,1,3],
            3 : [0,1,2],
            },
        )

g5 = Graph(number_of_vertices=5, directed=False,
        adjacency_dict = {
            0 : [1,2,3,4],
            1 : [0,2,3,4],
            2 : [0,1,3,4],
            3 : [0,1,2,4],
            4 : [0,1,2,3],
            },
        )

g6 = Graph(number_of_vertices=6, directed=False,
        adjacency_dict = {
            0 : [1,2,3,4,5],
            1 : [0,2,3,4,5],
            2 : [0,1,3,4,5],
            3 : [0,1,2,4,5],
            4 : [0,1,2,3,5],
            5 : [0,1,2,3,4],
            },
        )

g7 = Graph(number_of_vertices=7, directed=False,
        adjacency_dict = {
            0 : [1,2,3,4,5,6],
            1 : [0,2,3,4,5,6],
            2 : [0,1,3,4,5,6],
            3 : [0,1,2,4,5,6],
            4 : [0,1,2,3,5,6],
            5 : [0,1,2,3,4,6],
            6 : [0,1,2,3,4,5]
            },
        )

g8 = Graph(number_of_vertices=8, directed=False,
        adjacency_dict = {
            0 : [1,2,3,4,5,6,7],
            1 : [0,2,3,4,5,6,7],
            2 : [0,1,3,4,5,6,7],
            3 : [0,1,2,4,5,6,7],
            4 : [0,1,2,3,5,6,7],
            5 : [0,1,2,3,4,6,7],
            6 : [0,1,2,3,4,5,7],
            7 : [0,1,2,3,4,5,6],
            },
        )

g9 = Graph(number_of_vertices=9, directed=False,
        adjacency_dict = {
            0 : [1,2,3,4,5,6,7,8],
            1 : [0,2,3,4,5,6,7,8],
            2 : [0,1,3,4,5,6,7,8],
            3 : [0,1,2,4,5,6,7,8],
            4 : [0,1,2,3,5,6,7,8],
            5 : [0,1,2,3,4,6,7,8],
            6 : [0,1,2,3,4,5,7,8],
            7 : [0,1,2,3,4,5,6,8],
            8 : [0,1,2,3,4,5,6,7]
            },
        )

g10 = Graph(number_of_vertices=10, directed=False,
        adjacency_dict = {
            0 : [1,2,3,4,5,6,7,8,9],
            1 : [0,2,3,4,5,6,7,8,9],
            2 : [0,1,3,4,5,6,7,8,9],
            3 : [0,1,2,4,5,6,7,8,9],
            4 : [0,1,2,3,5,6,7,8,9],
            5 : [0,1,2,3,4,6,7,8,9],
            6 : [0,1,2,3,4,5,7,8,9],
            7 : [0,1,2,3,4,5,6,8,9],
            8 : [0,1,2,3,4,5,6,7,9],
            9 : [0,1,2,3,4,5,6,7,8],
            },
        )


# special graphs

g3_a = Graph(number_of_vertices=3, directed=False,
        adjacency_dict = {
            0 : [1,2],
            1 : [0],
            2 : [0],
            },
        )

g3_b = Graph(number_of_vertices=3, directed=False,
        adjacency_dict = {
            0 : [2],
            1 : [2],
            2 : [0, 1],
            },
        )

#print("iso? g3, g3_a : {}".format(isomorphic(g3, g3_a)))
#print("iso? g3_a, g3_b : {}".format(isomorphic(g3_a, g3_b)))

g4_a = Graph(number_of_vertices=4, directed=False,
          adjacency_dict = {
              0: [1, 3],
              1: [0, 3],
              2: [3],
              3: [0, 1, 2],
              }
          )


    
g_q = Graph(number_of_vertices=5, directed=False,
        adjacency_dict = {
            3: [0],
            4: [0],
            1: [2],
            2: [1],
            0: [3, 4],
            },
        vertex_coloring = [
            ],
        )

g_q_1 = Graph(number_of_vertices=5, directed=False,
        adjacency_dict = {
            1: [2],
            2: [3],
            3: [1],
            4: [0],
            0: [4],
            },
        vertex_coloring = [
            ],
        )

#print ("g_q disconnected? ", check_disconnected(g_q_1.adjacency_dict))

#node_count = 5
#automorphisms = deepcopy(generators)
#automorphisms.append([i for i in range(0, node_count)])
#print (automorphisms)

def permute(node_count, aut1, aut2):
    processed = []
    
    for i in range(0, node_count):
        if aut2[i] != i and i not in processed:
            tmp = aut1[aut2[i]]
            aut1[aut2[i]] = aut1[i] #aut2[aut2[i]]]
            aut1[i] = tmp
            processed.append(i)
            processed.append(aut2[i])
            
    return aut1

#print (permute(4, [0,2,1,3], [1,0,3,2]))


def aut_combinations(gen_count):
    lst_combinations = []
    
    length = gen_count
    lst = [i for i in range(0, gen_count)]
    
    while length > 1:
        cmbs = list(itertools.combinations(lst, length))
        for cmb in cmbs:
            perms = list(itertools.permutations(cmb))
            lst_combinations = lst_combinations + perms
        length = length - 1
        
    lst_combinations = list(OrderedDict.fromkeys(lst_combinations)) 
    
    return lst_combinations

#aut_cmbs = aut_combinations(3)
#print ("combinations : {} , {}".format(len(aut_cmbs), aut_cmbs))


def generate_automorphisms(node_count, gen, grpsize):
    automorphisms = [] #deepcopy(gen)
    #automorphisms.append([i for i in range(0, node_count)])
    
    print ("generators => {}".format(gen))
    print ("group size => {}".format(grpsize))
    
    p = []
    
    if len(gen) > 1:
        aut_cmbs = aut_combinations(len(gen))
    
        for cmb in aut_cmbs:
            lst = []
            for i in range(0, len(cmb)):
                lst.append(gen[i])
                p.append(lst)
    else:
        p.append(gen)
        
        
    i = 0
    k = 0
    m = 0
    n = 0
    gen_c = len(p[m])
    src = p[m][1]
    curr_ls = []
    
    while len(automorphisms) < grpsize: # and k < 20000:
        #if flag:
            #print ("1 {} <=> {}".format(src, gen2))
        src_cp = deepcopy(src)
        gen_k = permute(node_count, src_cp, p[m][i % gen_c])
        if gen_k not in curr_ls:
            curr_ls.append(gen_k)
            if gen_k not in automorphisms:
                #print("-- appending {}".format(gen_k))
                automorphisms.append(gen_k)
                
                if len(automorphisms) == grpsize:
                    print ("{} : {} : {}".format(m, len(curr_ls), curr_ls))
        else:
            m = m + 1
            if m == len(p):
                print ("m is {}".format(len(p)))
                #break
                src = automorphisms[n]
                n = n + 1
                m = 0
                gen_c = len(p[m])
                curr_ls = []
                i = 0
                continue
                
            print ("{} : {} : {} : {}".format(m - 1, (i + 1) % len(p[m]), len(curr_ls), curr_ls))
            
            if n == 0:
                src = p[m][1]
            else:
                src = automorphisms[n]

            gen_c = len(p[m])
            curr_ls = []
            i = 0
            continue

        src = gen_k
        i = i + 1
        k = k + 1

    print ("k is {}".format(k))

    return automorphisms

(generators, grpsize1, grpsize2, orbits, num_orbits) = autgrp(g4)
#print ("gen: {}, grpsize : {}".format(generators, grpsize1))
#autm = generate_automorphisms(g4.number_of_vertices, generators, grpsize1)

def get_adjacency_matrix(adj_dict):
    
    nodes = list(adj_dict)
    nodes.sort()
    ln = len(nodes)
    adj_mtx = np.zeros(shape=(ln, ln))
    #flat_adj_matrix = []
    
    for n in nodes:
        #row = [0 for i in range(0, ln)]
        for v in adj_dict[n]:
            #row[v] = 1
            adj_mtx[n][v] = 1
        #adj_mtx = np.append(adj_mtx, row)
    
    return adj_mtx

#print(g4.adjacency_dict)
#print(get_adjacency_matrix(g4.adjacency_dict))
#print(get_adjacency_matrix(g4.adjacency_dict).flatten())

def get_laplacian_matrix(adj_dict):
    
    nodes = list(adj_dict)
    nodes.sort()
    ln = len(nodes)
    lap_mtx = np.zeros(shape=(ln, ln))
    
    for n in nodes:
        for v in adj_dict[n]:
            lap_mtx[n][v] = -1
            
        lap_mtx[n][n] = len(adj_dict[n])
    
    return lap_mtx

#print(g4_a.adjacency_dict)
#print(list(g4_a.adjacency_dict))
#print(get_laplacian_matrix(g4_a.adjacency_dict))
#print(get_laplacian_matrix(g4_a.adjacency_dict).flatten())


def generate_isomorphisms(graph, isomorphisms_adj):
    node_count = graph.number_of_vertices
    adj_dict = graph.adjacency_dict
    isomorphisms = []
    #isomorphisms_adj = []
    
    #print ("generate_isomorphisms -> ", isomorphisms_adj)
    
    idty = [i for i in range(0, node_count)]
    perms = list(itertools.permutations(idty))
    
    for p in perms:
        r_lst = list(map(lambda x, y: (x,y), idty, p))
        d = dict((x,y) for x, y in r_lst)
        #adj_d = {}
        graph_n = Graph(node_count)
        
        for k,v in adj_dict.items():
            #adj_d[d[k]] = [d[j] for j in v]
            graph_n.connect_vertex(d[k], [d[j] for j in v])

        #graph_n.adjacency_dict = adj_d

        adj_d = get_adjacency_matrix(graph_n.adjacency_dict)
        adj_d = adj_d.flatten()
        adj_m, = np.where((isomorphisms_adj == adj_d).all(axis=1))

        if adj_m.size == 0:  #Eliminating automorphisms
        
            if isomorphic(graph, graph_n):
            #adj_d = {}
            #for k,v in graph_n.adjacency_dict.items():
            #    adj_d[k] = set(v)
                isomorphisms.append(graph_n)
                isomorphisms_adj = np.append(isomorphisms_adj, [adj_d], axis=0)

    return (isomorphisms, isomorphisms_adj)

#iso = generate_isomorphisms(g3_a, [])

#print ("{} : {}".format(len(iso), iso))

def check_disconnected(adj_dict):
    components = []
    node_count = len(list(adj_dict))
    
    for k,v in adj_dict.items():
        v.append(k)
        new_component = set(v)
        components.append(new_component)
    
    k = len(components)
    #print ("components -> ", components)
    prev_cluster = components[0]
    
    while k > 0:
    
        cluster = reduce(lambda x, y: x.union(y) if not x.isdisjoint(y) else x, components)
        #print ("cluster ", cluster)
        components.pop(0)
        components.insert(0, cluster)
        k = k - 1
        
        if prev_cluster == cluster:
            break
        
        prev_cluster = cluster
    
    #print("c[0] ", len(components[0]))
    #print("node cnt ", node_count)
    if len(components[0]) == node_count:
        return False
    else:
        return True
    
    

def generate_graph_isomorphisms_helper(graph, edge_remove_list):
    node_count = graph.number_of_vertices
    #adj_dict = graph.adjacency_dict
    isomorphisms = [] #[graph]
    isomorphisms_adj = np.zeros(shape=(1, node_count * node_count))
    deg_dict = {}
    
    m = 0
    cls = 0
        
    for i in range(0, len(edge_remove_list)):
        
        isomorphisms_c = []
        
        for j in range(0, len(edge_remove_list[i])):
            new_graph = deepcopy(graph)
            adj_dict = new_graph.adjacency_dict
            graph_modified = False
            graph_disconnected = False
            #isomorphisms_adj = []
            
            if type(edge_remove_list[i][j][0]) != tuple:
                lst = []
                lst.append(edge_remove_list[i][j])
                edge_remove_list[i][j] = tuple(lst)
                
            for k in edge_remove_list[i][j]:
                (a, b) = k #edge_remove_list[i][j]
                
                nodes = list(adj_dict)
                
                if len(nodes) < node_count or [] in adj_dict.values(): # or check_disconnected(adj_dict_disc_chk):
                    graph_disconnected = True
                    break
                
                if b in adj_dict[a] and a in adj_dict[b]:
                    adj_dict[a].remove(b)
                    adj_dict[b].remove(a)
                    
                    #check if graph is disconnected
                    if adj_dict[a] == [] or adj_dict[b] == []:
                        graph_disconnected = True
                        break
                    
                    graph_modified = True
                    
            if graph_modified and not graph_disconnected:
                
                adj_dict_disc_chk = deepcopy(new_graph.adjacency_dict)
                if not check_disconnected(adj_dict_disc_chk):
                
                    iso_exists = False
                    
                    adj_d = get_adjacency_matrix(new_graph.adjacency_dict)
                    adj_d = adj_d.flatten()
                    adj_m, = np.where((isomorphisms_adj == adj_d).all(axis=1))

                    if adj_m.size == 0:
                        iso, isomorphisms_adj = generate_isomorphisms(new_graph, isomorphisms_adj)
                        
                        if iso != []:
                            print("class: {}, {}".format(cls, iso[0]))
                            cls = cls + 1
                            isomorphisms_c.append(iso)
                            
                            deg_lst = []
                            for k,v in iso[0].adjacency_dict.items():
                                deg_lst.append(len(v))
                            deg_lst.sort()
                            t = tuple(deg_lst)
                            if t not in deg_dict:
                                deg_dict[t] = [iso]
                            else:
                                deg_dict[t].append(iso)
            
            m = m + 1
        
        if isomorphisms_c != []:
            isomorphisms.append(isomorphisms_c)
        
    return isomorphisms, deg_dict
    
def generate_graph_isomorphisms(graph):
    node_count = graph.number_of_vertices
    idty = [i for i in range(0, node_count)]
    edges = list(itertools.combinations(idty, 2))
    
    edge_remove_list = []
    #edge_remove_list.append(edges)
    
    print ("count of edges: {}\n".format(len(edges)))
    
    #for i in range(2, len(edges)):
    #    edge_cmbs = list(itertools.combinations(edges, i))
    #    edge_remove_list.append(edge_cmbs)

    edge_cmbs = list(itertools.combinations(edges, 5))
    edge_remove_list.append(edge_cmbs)
           
    isms = generate_graph_isomorphisms_helper(graph, edge_remove_list)
    
    #print ("edge_remove_list_len: {}".format(len(edge_remove_list)))
    
    return isms
    
iso, deg_dict = generate_graph_isomorphisms(g5)

#print ("\nisomorphisms degrees length: {}\n".format(len(deg_dict[(1, 2, 2, 2, 3)])))

print ("\nisomorphisms: {}\n".format(iso[0]))
#print ("\nisomorphisms length => {}".format(len(iso[0])))
#print ("\nisomorphisms degrees: {}\n".format(deg_dict.keys()))
#np_iso = np.array(iso[0])
#print ("\nisomorphisms repr: {}\n".format(np_iso))

def generate_test_data_ism(isomorphisms):
    #f = open("training_data.txt", "a")
    
    g1 = isomorphisms[0]
    test_data = []
    cls = 0
    
    #with open('test_data_g5.txt', 'w') as fh:
        #fh.writelines("{}:{}\n".format(cls, get_adjacency_matrix(g1.adjacency_dict).tolist()))
    test_data.append("{}:{}\n".format(cls, get_adjacency_matrix(g1.adjacency_dict).tolist()))
    cls = cls + 1
        #fh.writelines("---------------------------------------------------------------------\n")
        #cnt = 0
    for data in isomorphisms[1:]:
        cnt = 0
        for ism_g in data:
            for ism in ism_g:
                #fh.writelines("{}:{}\n".format(cls, get_adjacency_matrix(ism.adjacency_dict).tolist()))
                test_data.append("{}:{}\n".format(cls, get_adjacency_matrix(ism.adjacency_dict).tolist()))
            cls = cls + 1
            #fh.writelines("{} ---------------------------------------------------------------------\n".format(cnt))
    
    fh = open('test_data_g5.txt', 'w')
    test_data = "".join(test_data)
    fh.write(test_data)

#generate_test_data_ism(iso)

def generate_test_data_ism_part(isomorphisms, filename):
    #g1 = isomorphisms[0]
    cls = 0
    test_data = []
    
    #with open('test_data_g4.txt', 'w') as fh:
        #fh.writelines("{}:{}\n".format(cls, get_adjacency_matrix(g1.adjacency_dict).tolist()))
        #cls = cls + 1
        #fh.writelines("---------------------------------------------------------------------\n")
        #cnt = 0
    for data in isomorphisms:
        for ism in data:
                #fh.writelines("{}:{}\n".format(cls, get_adjacency_matrix(ism.adjacency_dict).tolist()))
            test_data.append("{}:{}\n".format(cls, get_adjacency_matrix(ism.adjacency_dict).tolist()))
        cls = cls + 1
            
    fh = open(filename, 'w')
    test_data = "".join(test_data)
    fh.write(test_data)

#generate_test_data_ism_part(iso[0])

def generate_test_data_ism_deg(deg_dict, basename):
    
    for deg,ism in deg_dict.items():
        deg_str = reduce(lambda x, y: str(x) + "_" + str(y), deg)
        
        filename = "".join((basename, "_", deg_str, "_.txt"))
        print(filename)
        generate_test_data_ism_part(ism, filename)

#generate_test_data_ism_deg(deg_dict, "test_data_g4_e5")

def generate_test_data_bij(isomorphisms):
    #f = open("training_data.txt", "a")
    
    g1 = isomorphisms[0]

    cls = list(g1.adjacency_dict)
    
    with open('test_data_bijection.txt', 'w') as fh:
        lm = get_laplacian_matrix(g1.adjacency_dict)
        
        for n in range(0, g1.number_of_vertices):
            lm_n = np.insert(lm, 0, lm[n], axis=0)
            fh.writelines("{}:{}\n".format(cls[n], lm_n.tolist()))
        #cls = cls + 1
        
        for ism in isomorphisms[1:]:
            #for ism_g in data:
                #for ism in data:
            lm = get_laplacian_matrix(ism.adjacency_dict)

            for n in range(0, ism.number_of_vertices):
                lm_n = np.insert(lm, 0, lm[n], axis=0)
                fh.writelines("{}:{}\n".format(cls[n], lm_n.tolist()))
            #cls = cls + 1

#print ("\n\nbijection test input: \n")
#print ("len: {}, data: {}\n".format(len(iso[1:][len(iso[1:]) - 1][1]), iso[1:][len(iso[1:]) - 1][1]))
#generate_test_data_bij(iso[1:][len(iso[1:]) - 1][1])

def generate_training_data_ism(isomorphisms):
    #f = open("training_data.txt", "a")
    cnt = 100
    #g1 = isomorphisms[0]

    cls = 0
    
    training_data = []
    
    #adj_m = get_adjacency_matrix(g1.adjacency_dict)
    #training_data.append((cls, adj_m.tolist()))
    y = 10
    
    #i = 0
    #while i < cnt:
    #    adj_m = get_adjacency_matrix(g1.adjacency_dict)
    #    d_n = np.where(adj_m==1, (cls+1) * y * random.random(), adj_m)
    #    training_data.append((cls, d_n.tolist()))
    #    i = i + 1
        
    #cls = cls + 1
    
    for data in isomorphisms:
            #for ism in data:
                for ism in data: #ism_g:
                    #fh.writelines("{}:{}\n".format(cls, get_adjacency_matrix(ism.adjacency_dict).tolist()))
                    i = 0
                    while i < cnt:
                        adj_m = get_adjacency_matrix(ism.adjacency_dict)
                        d_n = np.where(adj_m==1, (cls+1) * y * random.random(), adj_m)
                        training_data.append((cls, d_n.tolist()))
                        i = i + 1
                cls = cls + 1
    
    random.shuffle(training_data)
    
    with open('training_data_g5.txt', 'w') as fh:
        
        for val in training_data:
            (c, d) = val
            fh.writelines("{}:{}\n".format(c, d))

#generate_training_data_ism(iso[0])

def generate_training_data_ism_part(isomorphisms):
    #f = open("training_data.txt", "a")
    cnt = 100
    #g1 = isomorphisms[0]

    cls = 0
    
    training_data = []
    
    #adj_m = get_adjacency_matrix(g1.adjacency_dict)
    #training_data.append((cls, adj_m.tolist()))
    y = 10
    
    #i = 0
    #while i < cnt:
    #    adj_m = get_adjacency_matrix(g1.adjacency_dict)
    #    d_n = np.where(adj_m==1, (cls+1) * y * random.random(), adj_m)
    #    training_data.append((cls, d_n.tolist()))
    #    i = i + 1
        
    #cls = cls + 1
    
    for data in isomorphisms:
            #for ism_g in data:
                for ism in data:
                    #fh.writelines("{}:{}\n".format(cls, get_adjacency_matrix(ism.adjacency_dict).tolist()))
                    i = 0
                    while i < cnt:
                        adj_m = get_adjacency_matrix(ism.adjacency_dict)
                        d_n = np.where(adj_m==1, (cls+1) * y * random.random(), adj_m)
                        training_data.append((cls, d_n.tolist()))
                        i = i + 1
                cls = cls + 1
    
    random.shuffle(training_data)
    
    with open('training_data_6.txt', 'w') as fh:
        
        for val in training_data:
            (c, d) = val
            fh.writelines("{}:{}\n".format(c, d))

#generate_training_data_ism_part(iso[len(iso) - 2])

def generate_training_data_bij(isomorphisms):
    #f = open("training_data.txt", "a")
    cnt = 100
    g1 = isomorphisms[0]

    cls = list(g1.adjacency_dict)
    
    training_data = []
    
    lap_m = get_laplacian_matrix(g1.adjacency_dict)
    
    for n in range(0, g1.number_of_vertices):
        lm_n = np.insert(lap_m, 0, lap_m[n], axis=0)
        training_data.append((cls[n], lm_n.tolist()))
    y = 10
    
    i = 0
    while i < cnt:
        lap_m = get_laplacian_matrix(g1.adjacency_dict)
        
        for n in range(0, g1.number_of_vertices):
            d_n = np.where(lap_m==-1, (cls[n]+1) * y * random.random(), lap_m)
            lm_n = np.insert(d_n, 0, d_n[n], axis=0)
            training_data.append((cls[n], lm_n.tolist()))
        
        i = i + 1
        
    for ism in isomorphisms[1:]:
        i = 0
        while i < cnt:
            lap_m = get_laplacian_matrix(ism.adjacency_dict)
                        
            for n in range(0, ism.number_of_vertices):
                d_n = np.where(lap_m==-1, (cls[n]+1) * y * random.random(), lap_m)
                lm_n = np.insert(d_n, 0, d_n[n], axis=0)
                training_data.append((cls[n], lm_n.tolist()))

            i = i + 1
    
    random.shuffle(training_data)
    
    with open('training_data_bijection.txt', 'w') as fh:
        
        for val in training_data:
            (c, d) = val
            fh.writelines("{}:{}\n".format(c, d))

#generate_training_data_bij(iso[1:][len(iso[1:]) - 1][1])

def generate_training_data_from_test(in_file, out_file_cls, out_file_data):

    with open(in_file, 'r') as fh1:
        #with open(out_file, 'w') as fh2:
        #cls = []
        training_data = []
        data = None
        y = 10
        cnt = 100
        file_data = fh1.read().splitlines()
        
        for ln in file_data:
            d = ln.split(':')
            
            #if int(d[0]) in [0, 1, 3, 9, 10, 11]:
            #    cnt = 30
            #else:
            #    cnt = 10
            
            i = 0
            data = np.array([ast.literal_eval(d[1].rstrip())])
            while i < cnt:
                d_n = np.where(data==1, (int(d[0])+1) * y * random.random(), data)    
                training_data.append((int(d[0]), d_n)) #.tolist()))
                i = i + 1
    
    random.shuffle(training_data)    
    cls, data = zip(*training_data)
    
    np.save(out_file_cls, np.asarray(cls))
    np.save(out_file_data, np.asarray(data))
        
    #with open("training_data_s1.txt", 'w') as fh2:
    #    for val in training_data:
    #        (c, d) = val
    #        fh2.writelines("{}:{}\n".format(c, d))

#generate_training_data_from_test("test_data_g7_2_12_6_4.txt", "training_data_g7_2_12_6_4_cls.npy", "training_data_g7_2_12_6_4_data.npy")
#generate_training_data_from_test("test_data_g5.txt", "training_data_g5_cls.npy", "training_data_g5_data.npy")

def generate_test_data_from_test(in_file, out_file):

    with open(in_file, 'r') as fh1:
        #with open(out_file, 'w') as fh2:
        #cls = []
        training_data = []
        data = None
        y = 10
        cnt = 10
        cls = 0
        prev = None
        file_data = fh1.read().splitlines()
        
        for ln in file_data:
            d = ln.split(':')
            
            i = 0
            if prev is not None:
                if prev != d[0]:
                    cls = cls + 1
            
            training_data.append((cls, d[1])) #d_n.tolist()))
            prev = d[0]
            
    with open(out_file, 'w') as fh2:
        
        for val in training_data:
            (c, d) = val
            fh2.writelines("{}:{}".format(c, d))


#generate_test_data_from_test("test_data_g7_12_6_4.txt", "test_data_g7_2_12_6_4.txt")

def generate_test_data_np_from_test(in_file, out_file_cls, out_file_data):

    with open(in_file, 'r') as fh1:
        cls = []
        data = []
        file_data = fh1.read().splitlines()
        
        for ln in file_data:
            d = ln.split(':')
            cls.append(int(d[0]))
            data.append(np.array([ast.literal_eval(d[1].rstrip())])) #d[1])
            
    np.save(out_file_cls, np.asarray(cls))
    np.save(out_file_data, np.asarray(data))

#generate_test_data_np_from_test("test_data_g7_2_12_6_4.txt", "test_data_g7_12_6_4_cls.npy", "test_data_g7_12_6_4_data.npy")
#generate_test_data_np_from_test("test_data_g5.txt", "test_data_g5_cls.npy", "test_data_g5_data.npy")

def read_training_data_ism(filename):
    
    with open(filename, 'r') as fh:
        cls = []
        data = None
        for ln in fh:
            d = ln.split(':')
            cls.append(d[0])
            
            #print(d[1].rstrip())
            
            if data is None:
                data = np.array([ast.literal_eval(d[1].rstrip())])
                #print (data)
            else:
                data = np.append(data, np.array([ast.literal_eval(d[1].rstrip())]), axis = 0)
        print (data) #, len(cls))

#read_training_data('training_data.txt')

print ("Success !")

