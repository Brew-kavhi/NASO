import json

import networkx as nx
from django.db import models


class Graph(models.Model):
    graph_data = models.TextField()

    def save_to_iamge(self):
        raise NotImplementedError

    def get_graph(self):
        graph_data = json.loads(self.graph_data)
        graph = nx.node_link_graph(graph_data)
        return graph

    def load_from_graph(self, nx_graph):
        self.graph_data = json.dumps(nx.node_link_data(nx_graph))
