import dgl

from dgl.data import DGLDataset
from dgl.data.utils import load_graphs

class MSPDataset(DGLDataset):
    def __init__(self, graph_paths=None, pos_enc_dim=16):
        super(MSPDataset, self).__init__(name='Metabolic stability prediction dataset')

        self.pos_enc_dim = pos_enc_dim
        self.graphs = []
        self.labels = []
        self.FPs    = []

        self.names = []

        for graph_path in graph_paths:
            name = graph_path.split('/')[-1]
            self.names.append( name )

            self.graph, self.label = load_graphs(graph_path)
            self.graph = self.positional_encoding( self.graph[0] )

            self.graphs.append( self.graph )
            self.labels.append( self.label['label'] )
            self.FPs.append(  self.label['g_morgan'] )

        assert len(self.graphs) == len(self.names), 'Difference length of graphs & labels'


    def positional_encoding(self, g):
        pos_enc = dgl.random_walk_pe(g, self.pos_enc_dim)
        g.ndata['pos_enc'] = pos_enc
        return g

    def __getlabels__(self):
        return self.names

    def __getitem__(self, idx):
        return self.graphs[idx], self.FPs[idx], self.labels[idx], self.names[idx]

    def __len__(self):
        return len(self.names)