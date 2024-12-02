# Optimized version of the structural analysis script
import gravis as gv
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from itertools import combinations
import random
from collections import Counter
import community as community_louvain

# Import data
node_file = "data/marvel-unimodal-nodes.csv"
edges_file = "data/marvel-unimodal-edges.csv"

nodi = pd.read_csv(node_file)
archi = pd.read_csv(edges_file)

# Display sample data
print("Esempio di dati dei nodi:")
print(nodi.head())
print("\nEsempio di dati degli archi:")
print(archi.head())

# Create graph
G = nx.Graph()
for _, row in nodi.iterrows():
    G.add_node(row['Id'], label=row['Label'])
for _, row in archi.iterrows():
    G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

print(f"\nNumero di nodi: {G.number_of_nodes()}")
print(f"Numero di archi: {G.number_of_edges()}")

# Degree
degree = dict(G.degree())
print("\nGrado dei nodi:")
print(degree)

degree_centrality = nx.degree_centrality(G)
print("\nCentralità di grado:")
print({k: round(v, 2) for k, v in degree_centrality.items()})

betweenness_centrality = nx.betweenness_centrality(G)
print("\nCentralità di betweenness:")
print({k: round(v, 2) for k, v in betweenness_centrality.items()})

# Graph visualization
plt.figure(figsize=(20, 15))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=10)
plt.title("Visualizzazione del Grafo")
plt.show()

# Triads
def calcola_triadi(grafo):
    triadi = {}
    for nodi in combinations(grafo.nodes, 3):
        n_archi = grafo.subgraph(nodi).number_of_edges()
        triadi.setdefault(n_archi, []).append(nodi)
    return triadi

triadi = calcola_triadi(G)
print(f"\nNumero totale di triadi: {len(triadi.get(2, [])) + len(triadi.get(3, []))}")
print(f"Numero di triadi chiuse: {len(triadi.get(3, []))}")

coeff_clustering = nx.transitivity(G)
print(f"Coefficiente di clustering globale: {coeff_clustering:.4f}")

# Triads containing specific nodes
personaggio1 = "Iron Man / Tony Stark"
personaggio2 = "Captain America"
triadi_con_personaggi = [triade for triade in triadi.get(3, []) if personaggio1 in triade and personaggio2 in triade]

if triadi_con_personaggi:
    print(f"\nNumero di triadi chiuse con '{personaggio1}' e '{personaggio2}': {len(triadi_con_personaggi)}")
    triade_top = max(triadi_con_personaggi, key=lambda triade: max(G.degree(n) for n in triade if n not in {personaggio1, personaggio2}))
    terzo_nodo = [n for n in triade_top if n not in {personaggio1, personaggio2}][0]
    print(f"Triade con il terzo nodo più connesso: {triade_top}, Terzo nodo: {terzo_nodo}, Connessioni: {G.degree(terzo_nodo)}")
else:
    print(f"Nessuna triade chiusa trovata con '{personaggio1}' e '{personaggio2}'.")

# Cliques
cliques = list(nx.find_cliques(G))
print(f"\nNumero totale di clique: {len(cliques)}")
print(f"Dimensione della clique massima: {len(max(cliques, key=len))}")
size_distribution = Counter(len(clique) for clique in cliques)
print("\nDistribuzione delle dimensioni delle clique:")
for size, count in sorted(size_distribution.items()):
    print(f"- Dimensione {size}: {count}")

# Clique weights
def analizza_peso_clique(clique, grafo):
    archi = [(u, v) for u, v in combinations(clique, 2) if grafo.has_edge(u, v)]
    peso_totale = sum(grafo[u][v]['weight'] for u, v in archi)
    peso_medio = peso_totale / len(archi) if archi else 0
    return peso_totale, peso_medio

clique_pesi = [(clique, *analizza_peso_clique(clique, G)) for clique in cliques]
clique_peso_massimo = max(clique_pesi, key=lambda x: x[1])
print(f"\nClique con peso totale massimo: {clique_peso_massimo[0]}, Peso totale: {clique_peso_massimo[1]}")

# k-core analysis
k_core_max = max(nx.core_number(G).values())
print(f"\nValore massimo di k-core: {k_core_max}")
k_core_dist = Counter(nx.core_number(G).values())
print("\nDistribuzione dei nodi nei k-core:")
for k, count in sorted(k_core_dist.items()):
    print(f"- k = {k}: {count} nodi")

# k-core density
print("\nDensità per ogni valore di k:")
for k in sorted(k_core_dist.keys()):
    k_core_grafo = nx.k_core(G, k)
    print(f"- k = {k}: {nx.density(k_core_grafo):.4f}")

# Ego-network
ego_target = "Captain America"
ego_grafo = nx.ego_graph(G, ego_target)
print(f"\nEgo-network per '{ego_target}': Nodi: {ego_grafo.number_of_nodes()}, Archi: {ego_grafo.number_of_edges()}")

comunita = community_louvain.best_partition(ego_grafo)
print("\nComunità rilevate nell'ego-network:")
for label, nodi in Counter(comunita.values()).items():
    print(f"- Comunità {label}: {nodi} nodi")
