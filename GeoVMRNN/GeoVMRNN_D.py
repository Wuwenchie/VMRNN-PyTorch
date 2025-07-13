import torch
import torch.nn as nn
import torch.nn.functional as F
from my_tools import unfold_func, fold_func
from einops import rearrange
from timm.models.swin_transformer import PatchEmbed, PatchMerging
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from vmamba import VSSBlock, SS2D
from typing import Optional, Callable
from functools import partial


class VSB(VSSBlock):
    def __init__(
        self,
        hidden_dim: int = 0,
        input_resolution: tuple = (224, 224), 
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            input_resolution=input_resolution,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_resolution = input_resolution

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.ln_1(x)

        if hx is not None:
            hx = self.ln_1(hx)
            x = torch.cat((x, hx), dim=-1)
            x = self.linear(x)
        x = x.view(B, H, W, C) 

        x = self.drop_path(self.self_attention(x))
 
        x = x.view(B, H * W, C)
        x = shortcut + x

        return x


class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution, depth,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, d_state=16, **kwargs):
        super(VMRNNCell, self).__init__()

        self.VSBs = nn.ModuleList(
            VSB(hidden_dim=hidden_dim, input_resolution=input_resolution,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                norm_layer=norm_layer, attn_drop_rate=attn_drop,
                d_state=d_state, **kwargs)
            for i in range(depth))

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)
        else:
            hx, cx = hidden_states
        
        outputs = []
        for index, layer in enumerate(self.VSBs):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)
            else:
                x = layer(outputs[-1], None)
                outputs.append(x)
                
        o_t = outputs[-1]
        Ft = torch.sigmoid(o_t)
        cell = torch.tanh(o_t)
        Ct = Ft * (cx + cell)
        Ht = Ft * torch.tanh(Ct)

        return Ht, (Ht, Ct)


class GeoCNN(nn.Module):
    """地理空間特徵提取卷積模塊"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatialAttention(nn.Module):
    """空間注意力機制"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class TemporalFeatureAggregator(nn.Module):
    """時序特徵聚合模塊"""
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.temporal_conv = nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1)
        self.temporal_norm = nn.BatchNorm1d(hidden_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.feature_compress = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, temporal_features):
        B, T, C, H, W = temporal_features.shape
        
        features = temporal_features.permute(0, 3, 4, 2, 1).contiguous()
        features = features.view(B * H * W, C, T)
        
        temporal_feat = self.temporal_conv(features)
        temporal_feat = self.temporal_norm(temporal_feat)
        temporal_feat = F.relu(temporal_feat)
        
        pooled_feat = self.global_pool(temporal_feat).squeeze(-1)
        weights = self.feature_compress(pooled_feat)
        
        weighted_feat = temporal_feat * weights.unsqueeze(-1)
        aggregated = torch.mean(weighted_feat, dim=-1)
        
        aggregated = aggregated.view(B, H, W, -1).permute(0, 3, 1, 2)
        
        return aggregated


class PredictionHead(nn.Module):
    """預測頭"""
    def __init__(self, input_dim, cube_dim, output_length, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.cube_dim = cube_dim
        self.output_length = output_length
        self.num_layers = num_layers
        
        # 多層預測網路
        layers = []
        hidden_dim = input_dim // 2
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1))
            elif i == num_layers - 1:
                layers.append(nn.Conv2d(hidden_dim, cube_dim * output_length, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
        
        self.prediction_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.prediction_layers(x)


class GeoVMRNN(nn.Module):
    """改進的 GeoVMRNN 模型 - 支持地理patch參數"""
    def __init__(self, mypara):
        super().__init__()
        self.mypara = mypara
        self.device = mypara.device
        
        # 使用與 Geoformer 相同的 patch 參數
        if hasattr(mypara, 'patch_size'):
            self.patch_size = mypara.patch_size
        else:
            self.patch_size = (4, 4)  # 默認值
            
        # 計算cube_dim（與Geoformer保持一致）
        if self.mypara.needtauxy:
            self.cube_dim = (
                (mypara.input_channal + 2) * self.patch_size[0] * self.patch_size[1]
            )
            self.input_channels = mypara.input_channal + 2
        else:
            self.cube_dim = (
                mypara.input_channal * self.patch_size[0] * self.patch_size[1]
            )
            self.input_channels = mypara.input_channal
        
        # 使用地理patch參數
        if hasattr(mypara, 'H0') and hasattr(mypara, 'W0'):
            self.H0 = mypara.H0
            self.W0 = mypara.W0
            self.emb_spatial_size = mypara.emb_spatial_size
        else:
            # 根據地理範圍計算圖片尺寸
            if hasattr(mypara, 'lat_range') and hasattr(mypara, 'lon_range'):
                self.img_height = int(mypara.lat_range[1] - mypara.lat_range[0])
                self.img_width = int(mypara.lon_range[1] - mypara.lon_range[0])
            else:
                # 如果沒有地理範圍信息，使用默認值
                self.img_height = 224
                self.img_width = 224
            
            self.H0 = self.img_height // self.patch_size[0]
            self.W0 = self.img_width // self.patch_size[1]
            self.emb_spatial_size = self.H0 * self.W0
        
        # 地理空間特徵提取
        self.geo_cnn = nn.Sequential(
            GeoCNN(self.input_channels, 64),
            GeoCNN(64, 128),
            GeoCNN(128, 256),
            SpatialAttention(256)
        )
        
        # 嵌入維度
        self.embed_dim = 256
        
        # 創建高階 VMRNN Cell
        self.vmrnn_cell = VMRNNCell(
            hidden_dim=self.embed_dim,
            input_resolution=(self.H0, self.W0),  # 使用地理patch分辨率
            depth=2,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            d_state=16
        )
        
        # patch嵌入層：將特徵圖轉換為patch序列
        self.patch_project = nn.Conv2d(256, self.embed_dim, 
                                     kernel_size=self.patch_size, 
                                     stride=self.patch_size)
        
        # 將patch序列轉換回圖像格式
        self.patch_to_img = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=256,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # 時序特徵聚合
        self.temporal_aggregator = TemporalFeatureAggregator(
            feature_dim=256, 
            hidden_dim=256
        )
        
        # 特徵融合層
        self.fusion_conv = nn.Conv2d(256 + 256, 512, kernel_size=1)
        self.fusion_norm = nn.BatchNorm2d(512)
        
        # 預測頭 - 使用cube_dim
        self.prediction_head = PredictionHead(
            input_dim=512,
            cube_dim=self.cube_dim,
            output_length=mypara.output_length,
            num_layers=3
        )
        
    def forward(self, predictor, predictand=None, train=True):
        """
        Args:
            predictor: (batch, input_length, C, H, W)
            predictand: (batch, output_length, C, H, W) - 訓練時使用
        Returns:
            outvar_pred: (batch, output_length, C, H, W)
        """
        batch_size, seq_len, C, H, W = predictor.shape
        
        # 初始化VMRNN隱藏狀態
        hidden_states = None
        
        geo_features = []
        vmrnn_features = []
        
        for t in range(seq_len):
            # 1. 地理空間特徵提取
            geo_feat = self.geo_cnn(predictor[:, t])  # [B, 256, H, W]
            geo_features.append(geo_feat)
            
            # 2. 將特徵轉換為patch序列
            patch_feat = self.patch_project(geo_feat)  # [B, embed_dim, H0, W0]
            B, C_embed, H_patch, W_patch = patch_feat.shape
            patch_embed_feat = patch_feat.view(B, C_embed, H_patch * W_patch).permute(0, 2, 1)
            
            # 3. 使用高階VMRNN Cell處理
            vmrnn_out, hidden_states = self.vmrnn_cell(patch_embed_feat, hidden_states)
            
            # 4. 將patch序列轉換回圖像格式
            vmrnn_out_img = vmrnn_out.permute(0, 2, 1).view(B, self.embed_dim, H_patch, W_patch)
            
            # 上採樣回原始尺寸
            vmrnn_feat = self.patch_to_img(vmrnn_out_img)  # [B, 256, H, W]
            vmrnn_features.append(vmrnn_feat)
        
        # 5. 時序特徵聚合
        geo_features_tensor = torch.stack(geo_features, dim=1)
        vmrnn_features_tensor = torch.stack(vmrnn_features, dim=1)
        
        aggregated_geo = self.temporal_aggregator(geo_features_tensor)
        aggregated_vmrnn = self.temporal_aggregator(vmrnn_features_tensor)
        
        # 6. 特徵融合
        fused_features = torch.cat([aggregated_geo, aggregated_vmrnn], dim=1)
        fused_features = F.relu(self.fusion_norm(self.fusion_conv(fused_features)))
        
        # 7. 預測
        prediction = self.prediction_head(fused_features)  # [B, cube_dim*output_length, H0, W0]
        
        # 8. 重塑並使用fold_func恢復原始尺寸（與Geoformer保持一致）
        prediction = prediction.view(
            batch_size, self.cube_dim, self.mypara.output_length, self.H0, self.W0
        ).permute(0, 2, 1, 3, 4)  # [B, output_length, cube_dim, H0, W0]
        
        # 使用fold_func恢復到原始尺寸
        outvar_pred = fold_func(
            prediction, output_size=(H, W), kernel_size=self.patch_size
        )
        
        return outvar_pred
    
    def predict(self, predictor):
        """推理模式"""
        return self.forward(predictor, train=False)


# 工廠函數
def create_geo_vmrnn(mypara):
    """創建 GeoVMRNN 模型的工廠函數"""
    return GeoVMRNN(mypara)
