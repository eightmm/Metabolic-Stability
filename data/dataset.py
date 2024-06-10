import dgl

from dgl.data import DGLDataset
from dgl.data.utils import load_graphs

class MSPDataset(DGLDataset):
    def __init__(self, paths, pos_dim=20):
        super(MSPDataset, self).__init__(name='Metabolic stability prediction dataset')

        self.paths = paths
        self.pos_dim = pos_dim

    def get_rwpe(self, g):
        g.ndata['pos_enc'] = dgl.random_walk_pe(g, self.pos_dim)
        return g

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = path.split('/')[-1]
        g, label = load_graphs( path )
        g, label = g[0], label['y']

        g = self.get_rwpe( g )

        return g, label, name

    def __len__(self):
        return len(self.paths)