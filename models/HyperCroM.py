import torch
import torch.nn as nn
from utils.tensor_ops import cus_sample, upsample_add
from models.Modules import BasicConv2d, DenseFusion, FeatureFusionAndPrediction
from models.HypergraphComputation import CrossModalHypergraphPerception
from models.pvt_v2_encoders import pvt_v2_b3



class HyperCroM(nn.Module):
    def __init__(self, construct_method='threshold', k_interact=4, thresh_interact=0.8):
        super(HyperCroM, self).__init__()
        
        self.upsample_add = upsample_add   
        self.upsample = cus_sample      
        
        # -------------------- Backbone (PVT) --------------------
        self.pvt_backbone_rgb = pvt_v2_b3 (pretrained="pretrained_weights/pvt_v2_b3.pth")
        self.pvt_backbone_t   = pvt_v2_b3 (pretrained="pretrained_weights/pvt_v2_b3.pth")
        
        # Adjust the number of output channels
        self.trans_rgb_32x = BasicConv2d(512, 64, 1)
        self.trans_rgb_16x = BasicConv2d(320, 64, 1)
        self.trans_rgb_8x  = BasicConv2d(128, 64, 1)
        self.trans_rgb_4x  = BasicConv2d(64,  64, 1)
        
        self.trans_t_32x = BasicConv2d(512, 64, 1)
        self.trans_t_16x = BasicConv2d(320, 64, 1)
        self.trans_t_8x  = BasicConv2d(128, 64, 1)
        self.trans_t_4x  = BasicConv2d(64,  64, 1)
        
        # ------------ Initial Fusion of Two Modalities ------------
        self.dense32x = DenseFusion(512, 64)    
        self.dense16x = DenseFusion(320, 64)
        self.dense8x  = DenseFusion(128, 64)
        self.dense4x  = DenseFusion(64,  64)
        
        # ------------ Cross-Modal Hypergraph Perception (CroM-HP) ------------
        
        self.hyper_comp_32x = CrossModalHypergraphPerception(
            in_channels=64, ds_scale=1, k_interact=k_interact, thresh_interact=thresh_interact, construct_method=construct_method
            )
        self.hyper_comp_16x = CrossModalHypergraphPerception(
            in_channels=64, ds_scale=2, k_interact=k_interact, thresh_interact=thresh_interact, construct_method=construct_method
            )
        self.hyper_comp_8x  = CrossModalHypergraphPerception(
            in_channels=64, ds_scale=4, k_interact=k_interact, thresh_interact=thresh_interact, construct_method=construct_method
            )
        self.hyper_comp_4x  = CrossModalHypergraphPerception(
            in_channels=64, ds_scale=8, k_interact=k_interact, thresh_interact=thresh_interact, construct_method=construct_method
            )
        
        # -------------------- Feature Fusion and Prediction --------------------
        self.fdm = FeatureFusionAndPrediction()  
    
    def forward(self, rgbt_inputs):

        rgb, t_img = rgbt_inputs[0], rgbt_inputs[1]
        
        # --------------- 1. Backbone Feature Extraction ---------------
        # (1) RGB Branch
        rgb_4x  = self.pvt_backbone_rgb.forward_stage1(rgb)
        rgb_8x  = self.pvt_backbone_rgb.forward_stage2(rgb_4x)
        rgb_16x = self.pvt_backbone_rgb.forward_stage3(rgb_8x)
        rgb_32x = self.pvt_backbone_rgb.forward_stage4(rgb_16x)
        
        # (2) T Branch
        t_4x  = self.pvt_backbone_t.forward_stage1(t_img)
        t_8x  = self.pvt_backbone_t.forward_stage2(t_4x)
        t_16x = self.pvt_backbone_t.forward_stage3(t_8x)
        t_32x = self.pvt_backbone_t.forward_stage4(t_16x)
        
        # Channel Adjustment
        trans_rgb_4x  = self.trans_rgb_4x(rgb_4x)
        trans_rgb_8x  = self.trans_rgb_8x(rgb_8x)
        trans_rgb_16x = self.trans_rgb_16x(rgb_16x)
        trans_rgb_32x = self.trans_rgb_32x(rgb_32x)
        
        trans_t_4x  = self.trans_t_4x(t_4x)
        trans_t_8x  = self.trans_t_8x(t_8x)
        trans_t_16x = self.trans_t_16x(t_16x)
        trans_t_32x = self.trans_t_32x(t_32x)
        
        # --------------- 2. Hypergraph Learning ---------------
        
        # Initial Fusion
        initial_fused_4x = self.dense4x(rgb_4x, t_4x)
        initial_fused_8x = self.dense8x(rgb_8x, t_8x)
        initial_fused_16x = self.dense16x(rgb_16x, t_16x)
        initial_fused_32x = self.dense32x(rgb_32x, t_32x)
        
        # CroM-HPs with pyramid structure
        fused_32x, _, _ = self.hyper_comp_32x(initial_fused_32x, trans_rgb_32x, trans_t_32x)
        
        initial_fused_16x = self.upsample_add(fused_32x,initial_fused_16x)
  
        fused_16x, _, _ = self.hyper_comp_16x(initial_fused_16x, trans_rgb_16x, trans_t_16x)
        
        initial_fused_8x = self.upsample_add(fused_16x, initial_fused_8x)
    
        fused_8x, _, _ = self.hyper_comp_8x(initial_fused_8x, trans_rgb_8x, trans_t_8x)
        
        initial_fused_4x = self.upsample_add(fused_8x, initial_fused_4x)
       
        fused_4x, _, _ = self.hyper_comp_4x(initial_fused_4x, trans_rgb_4x, trans_t_4x)
        
        
        # --------------- 3. Feature Fusion and Prediction ---------------
        final_pred = self.fdm(fused_4x, fused_8x, fused_16x, fused_32x)
         
        return final_pred

def fusion_model(construct_method='threshold', k_interact=4, thresh_interact=0.8):
    model = HyperCroM(construct_method, k_interact, thresh_interact)
    return model
        

