import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Modules import BasicConv2d
from models.Modules import CBAM


class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=(1, 2, 3, 4)):
        super(DilatedConv, self).__init__()
        
        self.conv_rgb = BasicConv2d(in_planes=in_channels, out_planes=out_channels, kernel_size=1)
        self.conv_t = BasicConv2d(in_planes=in_channels, out_planes=out_channels, kernel_size=1)
        self.conv_fuse = BasicConv2d(in_planes=in_channels, out_planes=out_channels, kernel_size=1)
        
        self.dilated_convs = nn.ModuleList([
            BasicConv2d(out_channels * 3, out_channels, kernel_size=3, dilation=rate, padding=rate)
            for rate in dilation_rates
        ])
        self.cbams = nn.ModuleList([
            CBAM(out_channels) for _ in range(len(dilation_rates))
        ])
        
        self.fusion = BasicConv2d(out_channels * len(dilation_rates), out_channels, kernel_size=1)
        self.residual = BasicConv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, rgb_input, t_input, fuse_input):

        trans_rgb = self.conv_rgb(rgb_input)
        trans_t = self.conv_t(t_input)
        trans_fuse = self.conv_fuse(fuse_input)
        
        concatenated_features = torch.cat([trans_rgb,trans_t,trans_fuse], dim=1)
        
        dilated_outputs = [conv(concatenated_features) for conv in self.dilated_convs]
        dilated_outputs = [cbam(dilated_output) for cbam, dilated_output in zip(self.cbams, dilated_outputs)]
        
        fused_features = torch.cat(dilated_outputs, dim=1)
        fused_features = self.fusion(fused_features)
        
        output = fused_features + self.residual(trans_fuse)
        
        return output  


def build_interaction_hyperedges_knn(
    X_target: torch.Tensor,
    X_context: torch.Tensor,
    k: int = 4
) -> torch.Tensor:

    N_t = X_target.size(0)
    N_c = X_context.size(0)

    distance_mat = torch.cdist(X_target, X_context)  # [N_t, N_c]
    _, knn_idx = torch.topk(distance_mat, k=k, dim=1, largest=False)  # [N_t, k]

    H = torch.zeros(N_t + N_c, N_t, dtype=torch.float32, device=X_target.device)

    for i in range(N_t):
        H[i, i] = 1.0
        selected_context_nodes = knn_idx[i] + N_t
        H[selected_context_nodes, i] = 1.0

    return H

def build_interaction_hyperedges_thresh(
    X_target: torch.Tensor,
    X_context: torch.Tensor,
    thresh: float = 0.5
) -> torch.Tensor:
   
    N_t = X_target.size(0)
    N_c = X_context.size(0)

    sim_mat = F.cosine_similarity(
        X_target.unsqueeze(1),  
        X_context.unsqueeze(0), 
        dim=2
    )  

    H = torch.zeros(N_t + N_c, N_t, dtype=torch.float32, device=X_target.device)

    for i in range(N_t):
        H[i, i] = 1.0
        selected_ctx = torch.nonzero(sim_mat[i] > thresh, as_tuple=False).flatten()
        H[selected_ctx + N_t, i] = 1.0

    return H


class HypergraphConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.node2edge_transform = nn.Linear(in_channels, out_channels, bias=True)
        self.edge2node_transform = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, X, H):
        """
        X: [V, Fin]
        H: [V, E]
        """
        # 节点->超边
        X_node_trans = self.node2edge_transform(X)  # [V, out_channels]
        X_edge = H.T @ X_node_trans                 # [E, out_channels]

        deg_e = H.sum(dim=0, keepdim=True).clamp_min(1.0)
        X_edge = X_edge / deg_e.T

        # 超边->节点
        X_edge_trans = self.edge2node_transform(X_edge)  # [E, out_channels]
        X_node_new = H @ X_edge_trans                    # [V, out_channels]

        deg_v = H.sum(dim=1, keepdim=True).clamp_min(1.0)
        X_node_new = X_node_new / deg_v

        return X_node_new

class HypergraphComputation(nn.Module):

    def __init__(self, F_dim, k_interact=5, thresh_interact=0.8, construct_method='threshold'):
        super().__init__()
        self.k_interact = k_interact
        self.thresh_interact = thresh_interact
        self.construct_method = construct_method
        if self.construct_method.lower() not in ['knn', 'threshold']:
            raise ValueError("construct_method must be 'knn' or 'threshold'.")
        
        self.W_g = nn.Parameter(torch.randn(F_dim, F_dim))

        self.hyperconv = HypergraphConv(F_dim, F_dim)

    def forward(self, X_target, X_context1, X_context2):
        """
        X_target:   [B, C, H, W]
        X_context1: [B, C, H, W]
        X_context2: [B, C, H, W]
        """
        B, C, H, W = X_target.shape
        device = X_target.device

        X_target_resh = (X_target.permute(0,2,3,1)
                                  .reshape(B, -1, C)
                                  .contiguous())
        X_context1_resh = (X_context1.permute(0,2,3,1)
                                     .reshape(B, -1, C)
                                     .contiguous())
        X_context2_resh = (X_context2.permute(0,2,3,1)
                                     .reshape(B, -1, C)
                                     .contiguous())
        N  = X_target_resh.shape[1]  

        X_context_resh = torch.cat([X_context1_resh, X_context2_resh], dim=1)
        N_ctx = X_context_resh.shape[1]  

        X_target_big  = X_target_resh.view(B*N, C)
        X_context_big = X_context_resh.view(B*N_ctx, C)

        total_rows = B * (N + N_ctx)
        total_cols = B * N
        H_big = torch.zeros(total_rows, total_cols, dtype=torch.float32, device=device)

        for i in range(B):
            t_start = i*N
            t_end   = t_start + N
            c_start = i*N_ctx
            c_end   = c_start + N_ctx
            X_t_i = X_target_big[t_start:t_end, :]   
            X_c_i = X_context_big[c_start:c_end, :]  

            if self.construct_method == 'knn':
                H_i = build_interaction_hyperedges_knn(X_t_i, X_c_i, k=self.k_interact)
            elif self.construct_method == 'threshold':
                H_i = build_interaction_hyperedges_thresh(X_t_i, X_c_i, thresh=self.thresh_interact)

            row_start = i*(N + N_ctx)
            row_end   = row_start + (N + N_ctx)
            col_start = i*N
            col_end   = col_start + N

            H_big[row_start:row_end, col_start:col_end] = H_i

        X_all_big = torch.cat([X_target_big, X_context_big], dim=0)  
        X_all_new_big = self.hyperconv(X_all_big, H_big)        

        X_target_out_big  = X_all_new_big[: B*N, :]
        X_context_out_big = X_all_new_big[B*N : , :]

        X_target_out_resh  = X_target_out_big.view(B, N,   C)
        X_context_out_resh = X_context_out_big.view(B, N_ctx, C)

        X_context1_out_resh = X_context_out_resh[:, :N, :]
        X_context2_out_resh = X_context_out_resh[:, N:, :]

        X_target_out = (X_target_out_resh.view(B, H, W, C)
                                        .permute(0,3,1,2)
                                        .contiguous())
        X_context1_out = (X_context1_out_resh.view(B, H, W, C)
                                           .permute(0,3,1,2)
                                           .contiguous())
        X_context2_out = (X_context2_out_resh.view(B, H, W, C)
                                           .permute(0,3,1,2)
                                           .contiguous())

        return X_target_out, X_context1_out, X_context2_out


class DownSampler(nn.Module):
    """
    ds_scale=1 => nn.Identity(), 
    ds_scale>1 => Conv2d(stride=ds_scale)
    """
    def __init__(self, in_channels, ds_scale=1):
        super().__init__()
        if ds_scale == 1:
            self.down = nn.Identity()
        else:
            self.down = nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=ds_scale, padding=1
            )
    def forward(self, x):
        return self.down(x)

class UpSampler(nn.Module):
    def __init__(self, in_channels, ds_scale=1):
        super().__init__()
        if ds_scale == 1:
            self.up = BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=ds_scale, mode='bilinear', align_corners=False),
                BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
    def forward(self, x):
        return self.up(x)
   
   
class CrossModalHypergraphPerception(nn.Module):

    def __init__(
        self,
        in_channels,   
        ds_scale=1,
        k_interact=3,
        thresh_interact=0.8,
        construct_method='knn'
    ):
        super().__init__()
        self.ds_scale = ds_scale
        
        self.dconv = DilatedConv(in_channels, in_channels)

        self.down_t = DownSampler(in_channels, ds_scale)
        self.down_c1= DownSampler(in_channels, ds_scale)
        self.down_c2= DownSampler(in_channels, ds_scale)

        self.hypergraph_one_shot = HypergraphComputation(
            F_dim=in_channels, 
            k_interact=k_interact,
            thresh_interact=thresh_interact,
            construct_method=construct_method
        )

        self.up_t  = UpSampler(in_channels, ds_scale)
        self.up_c1 = UpSampler(in_channels, ds_scale)
        self.up_c2 = UpSampler(in_channels, ds_scale)
        
        self.fusion = nn.Sequential(
            BasicConv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
            BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            BasicConv2d(in_channels, in_channels, kernel_size=1, padding=0),
            CBAM(in_channels)           
        )
        
        

    def forward(self, X_target, X_context1, X_context2):
        """
        X_target, X_context1, X_context2: [B, C, H, W], shape一致
        返回: 同shape的 (X_target_out, X_context1_out, X_context2_out)
        """
        
        X_target = self.dconv(X_context1, X_context2, X_target)
        
        X_target_ds   = self.down_t(X_target)
        X_context1_ds = self.down_c1(X_context1)
        X_context2_ds = self.down_c2(X_context2)

        X_t_out_ds, X_c1_out_ds, X_c2_out_ds = self.hypergraph_one_shot(
            X_target_ds, X_context1_ds, X_context2_ds
        )

        X_t_out   = self.up_t(X_t_out_ds)
        X_c1_out  = self.up_c1(X_c1_out_ds)
        X_c2_out  = self.up_c2(X_c2_out_ds)
    
        X_t_out = self.fusion(torch.cat([X_t_out, X_target], dim=1))
        
        return X_t_out, X_c1_out, X_c2_out




if __name__ == "__main__":
    torch.manual_seed(0)

    B = 16
    C = 64
    H, W = 16, 16  
    ds = 1

    X_t   = torch.randn(B, C, H, W)
    X_c1  = torch.randn(B, C, H, W)
    X_c2  = torch.randn(B, C, H, W)

    model = CrossModalHypergraphPerception(
        in_channels=C,
        ds_scale=ds,
        thresh_interact=0.8,
        construct_method='threshold'
    )

    X_t_out, X_c1_out, X_c2_out = model(X_t, X_c1, X_c2)

    print("X_t_out.shape  =", X_t_out.shape)   # [B, C, H, W]
    print("X_c1_out.shape =", X_c1_out.shape)
    print("X_c2_out.shape =", X_c2_out.shape)

    print('Model Params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

