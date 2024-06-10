import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling

import torchbnn as bnn

from .GatedGCNLSPE import GatedGCNLSPELayer
from .GraphTransformer import GraphTransformerLayer

class MetabolicStabilityPrediction(nn.Module):
    def __init__(self, in_dim, emb_dim, edge_dim, pose_dim, num_layers, dropout=0.2):
        super(MetabolicStabilityPrediction, self).__init__()
        self.node_encoder = nn.Linear( in_dim,   emb_dim )
        self.edge_encoder = nn.Linear( edge_dim, emb_dim )
        self.pose_encoder = nn.Linear( pose_dim, emb_dim )

        self.node_norm = nn.LayerNorm( emb_dim )
        self.edge_norm = nn.LayerNorm( emb_dim )

        self.gated_gcn_layers = nn.ModuleList(
            [
                GatedGCNLSPELayer(
                    input_dim=emb_dim,
                    output_dim=emb_dim,
                    dropout=dropout,
                    batch_norm=True,
                    use_lapeig_loss=False,
                    residual=True
                )
                for _ in range(num_layers)
            ]
        )

        self.GT_layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    emb_dim,
                    residual=True,
                    num_attention_heads=8,
                    dropout_rate=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.MLP = nn.Sequential(
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(p=dropout),
            nn.Linear(emb_dim, 1),
        )

        self.weight1 = nn.Parameter( th.tensor( 0.5 ) )
        self.weight2 = nn.Parameter( th.tensor( 0.5 ) )
        self.pooling = SumPooling()

    def forward(self, g):
        h = self.node_encoder( g.ndata['feats'] )
        e = self.edge_encoder( g.edata['feats'] )
        p = self.pose_encoder( g.ndata['pos_enc'] )

        h = self.node_norm( h )
        e = self.edge_norm( e )

        h_i = h
        e_i = e
        p_i = p

        for gated_gcn_layer, gt_layer in zip( self.gated_gcn_layers, self.GT_layers ):
            h1, p, e1 = gated_gcn_layer(g, h, p, e, 1)
            h2,    e2 = gt_layer(g, h, e)
            h = ( h1 * self.weight1 ) + ( h2 * self.weight2 )
            e = ( e1 * self.weight1 ) + ( e2 * self.weight2 )
            h += h_i
            e += e_i
            p += p_i

        h = self.pooling(g, h)
        h = self.MLP(h)

        return h


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

