import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling

import torchbnn as bnn

from .GatedGCNLSPE import GatedGCNLSPELayer
from .mha import MultiHeadAttention
from .mpmha import MPNN_MHA

class MetabolicStabilityPrediction(nn.Module):
    def __init__(self, in_dim, emb_dim, edge_dim, pose_dim, fp_dim, num_layers=4):
        super(MetabolicStabilityPrediction, self).__init__()
        self.node_encoder = nn.Linear( in_dim,   emb_dim )
        self.edge_encoder = nn.Linear( edge_dim, emb_dim )
        self.pose_encoder = nn.Linear( pose_dim, emb_dim )

        self.node_norm = nn.LayerNorm( emb_dim )
        self.edge_norm = nn.LayerNorm( emb_dim )

        self.gnn_layers = nn.ModuleList(
            [
                GatedGCNLSPELayer(
                    input_dim=emb_dim,
                    output_dim=emb_dim,
                    dropout=0.1,
                    batch_norm=True,
                    use_lapeig_loss=False,
                    residual=True
                )
                for _ in range(num_layers)
            ]
        )

        self.MLP = nn.Sequential(nn.Linear(emb_dim, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ELU(),
                                 nn.Dropout(p=0.1),
                                 nn.Linear(128, 32),
                                 nn.BatchNorm1d(32),
                                 nn.ELU(),
                                 nn.Dropout(p=0.1),
                                 nn.Linear(32, 1),
                                )

        self.pooling = SumPooling()

    def forward(self, g, fp):
        h = self.node_encoder( g.ndata['feats'] )
        e = self.edge_encoder( g.edata['feats'] )
        p = self.pose_encoder( g.ndata['pos_enc'] )

        h = self.node_norm( h )
        e = self.edge_norm( e )

        h_i = h
        e_i = e
        p_i = p

        for layer in self.gnn_layers:
            h, p, e = layer(g, h, p, e, 1)
            h += h_i
            e += e_i
            p += p_i

        h = self.pooling(g, h)
        h = self.MLP(h)

        return h

# class MetabolicStabilityPrediction(nn.Module):
#     def __init__(self, in_dim, emb_dim, edge_dim, pose_dim, fp_dim, num_layers=4):
#         super(MetabolicStabilityPrediction, self).__init__()
#         self.node_encoder = nn.Linear( in_dim,   emb_dim )
#         self.edge_encoder = nn.Linear( edge_dim, emb_dim )
#         self.pose_encoder = nn.Linear( pose_dim, emb_dim )

#         self.node_norm = nn.LayerNorm( emb_dim )
#         self.edge_norm = nn.LayerNorm( emb_dim )

#         self.gps_layer = nn.ModuleList(
#             [
#                 MPNN_MHA(
#                     emb_dim,
#                     4
#                 )
#                 for _ in range(num_layers)
#             ]
#         )

#         self.MLP = nn.Sequential(nn.Linear(emb_dim * 2, 128),
#                                  nn.BatchNorm1d(128),
#                                  nn.ELU(),
#                                  nn.Dropout(p=0.1),
#                                  nn.Linear(128, 32),
#                                  nn.BatchNorm1d(32),
#                                  nn.ELU(),
#                                  nn.Dropout(p=0.1),
#                                  nn.Linear(32, 1),
#                                 )

#         self.bnn_mlp = nn.Sequential(
#             nn.Linear(emb_dim, emb_dim),
#             nn.BatchNorm1d(emb_dim),
#             nn.ELU(),
#             nn.Dropout(p=0.2),
#             bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=emb_dim, out_features=1),
#         )

#         self.fp_mlp = nn.Sequential(nn.Linear(fp_dim, emb_dim),
#                                  nn.BatchNorm1d(emb_dim),
#                                  nn.ELU(),
#                                  nn.Dropout(p=0.1),
#                                  nn.Linear(emb_dim, emb_dim),
#                                 )

#         self.pooling = SumPooling()

#     def forward(self, g, fp):
#         h = self.node_encoder( g.ndata['feats'] )
#         e = self.edge_encoder( g.edata['feats'] )
#         p = self.pose_encoder( g.ndata['pos_enc'] )

#         h = self.node_norm( h )
#         e = self.edge_norm( e )

#         h_i = h
#         e_i = e
#         p_i = p

#         for layer in self.gps_layer:
#             h, p, e = layer(g, h, p, e)

#         h = self.pooling(g, h)
#         # h = self.bnn_mlp(h)

#         h_fp = self.fp_mlp(fp)

#         h = th.concat( [h, h_fp], dim=1)

#         h = self.MLP(h)

#         return h


class MSP_MoE(nn.Module):
    def __init__(self, node_dim, edge_dim, pose_dim, emb_dim, num_layers, fp_dim, num_experts):
        super(MSP_MoE, self).__init__()

        self.expert_models = nn.ModuleList(
            [
                MetabolicStabilityPrediction( node_dim, emb_dim, edge_dim, pose_dim, fp_dim, num_layers=4 )
                for _ in range(num_experts)
            ]
        )

        self.gating_network = nn.Sequential(
            nn.Linear(fp_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(emb_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        self.pooling = SumPooling()

    def forward(self, g, fingerprint):
        gating_probs = self.gating_network(fingerprint)

        expert_outputs = [ expert(g, fingerprint) for expert in self.expert_models ]

        combined_output = th.concat(expert_outputs, dim=1)

        ensemble_output = th.sum(combined_output * gating_probs, dim=-1).unsqueeze(1)

        return ensemble_output

