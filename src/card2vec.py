import networkx as nx
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from node2vec import Node2Vec

with open('../dataset/hsreply_data/decks.json', 'r') as file:
    decks = json.load(file)

G = nx.Graph()

winrate = [deck['win_rate'] for deck in decks]
avg_winrate = sum(winrate) / len(winrate)
print("win_rate:", avg_winrate)

for deck in tqdm(decks):
    for card in deck['deck_list']:
        G.add_node(card[0])

    if deck['win_rate'] >= 1:
        for card1 in deck['deck_list']:
            for card2 in deck['deck_list']:
                if card1[0] == card2[0]:
                    continue
                G.add_edge(card1[0], card2[0])

print(len(G.nodes()))
print(len(G.edges()))
fig = plt.figure(figsize=(30, 30)) 
pos = nx.spring_layout(G)  # 选择布局算法;
nx.draw(G, pos, with_labels=True)
plt.axis('equal') 
plt.savefig('graph.png')
plt.show()

node2vec = Node2Vec(G, dimensions=512, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

model.wv.save_word2vec_format('emb_512.txt')