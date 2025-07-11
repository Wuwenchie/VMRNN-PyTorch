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


class PatchExpanding(nn.Module):
    r""" Patch Expanding Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                       stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.Conv(x)

        return x


class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution, depth,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, d_state=16, **kwargs):
        """
        Args:
        hidden_dim: Dimension of the hidden layer.
        input_resolution: Tuple of the input resolution.
        depth: Depth of the cell.
        drop, attn_drop, drop_path: Parameters for VSB.
        norm_layer: Normalization layer.
        d_state: State dimension for SS2D in VSB.
        """
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
    """空間注意力機制（保留，因為這是局部注意力，不是全局自注意力）"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B, C, H, W]
        attention = self.sigmoid(self.conv(x))
        return x * attention


class TemporalFeatureAggregator(nn.Module):
    """時序特徵聚合模塊 - 替代 Transformer 編碼器"""
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 時序特徵聚合
        self.temporal_conv = nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1)
        self.temporal_norm = nn.BatchNorm1d(hidden_dim)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特徵壓縮
        self.feature_compress = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, temporal_features):
        """
        Args:
            temporal_features: [B, T, C, H, W]
        Returns:
            aggregated_features: [B, C, H, W]
        """
        B, T, C, H, W = temporal_features.shape
        
        # 重塑為 [B*H*W, C, T] 進行時序卷積
        features = temporal_features.permute(0, 3, 4, 2, 1).contiguous()
        features = features.view(B * H * W, C, T)
        
        # 時序特徵聚合
        temporal_feat = self.temporal_conv(features)
        temporal_feat = self.temporal_norm(temporal_feat)
        temporal_feat = F.relu(temporal_feat)
        
        # 全局池化得到時序摘要
        pooled_feat = self.global_pool(temporal_feat).squeeze(-1)  # [B*H*W, hidden_dim]
        
        # 特徵權重
        weights = self.feature_compress(pooled_feat)  # [B*H*W, hidden_dim]
        
        # 加權聚合
        weighted_feat = temporal_feat * weights.unsqueeze(-1)
        aggregated = torch.mean(weighted_feat, dim=-1)  # [B*H*W, hidden_dim]
        
        # 重塑回空間維度
        aggregated = aggregated.view(B, H, W, -1).permute(0, 3, 1, 2)
        
        return aggregated


class PredictionHead(nn.Module):
    """預測頭 - 替代 Transformer 解碼器"""
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 多層預測網路
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1))
            elif i == num_layers - 1:
                layers.append(nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
        
        self.prediction_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.prediction_layers(x)


class GeoVMRNN(nn.Module):
    """使用高階VMRNN Cell的地理時空預測模型"""
    def __init__(self, mypara):
        super().__init__()
        self.mypara = mypara
        self.device = mypara.device
        
        # 計算輸入維度
        if self.mypara.needtauxy:
            self.input_channels = mypara.input_channal + 2
        else:
            self.input_channels = mypara.input_channal
        
        # 地理空間特徵提取
        self.geo_cnn = nn.Sequential(
            GeoCNN(self.input_channels, 64),
            GeoCNN(64, 128),
            GeoCNN(128, 256),
            SpatialAttention(256)
        )
        
        # 計算patch嵌入相關參數
        self.patch_size = 4
        self.embed_dim = 256
        
        # 使用左邊的高階VMRNN Cell
        # 假設輸入圖像尺寸為224x224，計算patch分辨率
        self.img_size = 224  # 根據實際情況調整
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, 
            patch_size=self.patch_size, 
            in_chans=256,  # 來自geo_cnn的輸出通道
            embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm
        )
        
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution
        
        # 創建高階VMRNN Cell
        self.vmrnn_cell = VMRNNCell(
            hidden_dim=self.embed_dim,
            input_resolution=patches_resolution,
            depth=2,  # 可調整
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            d_state=16
        )
        
        # 將patch序列轉換回圖像格式的層
        self.patch_to_img = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=256,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # 時序特徵聚合（替代Transformer編碼器）
        self.temporal_aggregator = TemporalFeatureAggregator(
            feature_dim=256, 
            hidden_dim=256
        )
        
        # 特徵融合層
        self.fusion_conv = nn.Conv2d(256 + 256, 512, kernel_size=1)
        self.fusion_norm = nn.BatchNorm2d(512)
        
        # 預測頭（替代Transformer解碼器）
        self.prediction_head = PredictionHead(
            input_dim=512,
            output_dim=self.input_channels * mypara.output_length,
            hidden_dim=256,
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
            patch_embed_feat = self.patch_embed(geo_feat)  # [B, L, embed_dim]
            
            # 3. 使用高階VMRNN Cell處理
            vmrnn_out, hidden_states = self.vmrnn_cell(patch_embed_feat, hidden_states)
            
            # 4. 將patch序列轉換回圖像格式
            B, L, embed_dim = vmrnn_out.shape
            H_patch, W_patch = self.patches_resolution
            vmrnn_out_img = vmrnn_out.view(B, H_patch, W_patch, embed_dim)
            vmrnn_out_img = vmrnn_out_img.permute(0, 3, 1, 2)  # [B, embed_dim, H_patch, W_patch]
            
            # 上採樣回原始尺寸
            vmrnn_feat = self.patch_to_img(vmrnn_out_img)  # [B, 256, H, W]
            vmrnn_features.append(vmrnn_feat)
        
        # 5. 時序特徵聚合
        geo_features_tensor = torch.stack(geo_features, dim=1)  # [B, T, 256, H, W]
        vmrnn_features_tensor = torch.stack(vmrnn_features, dim=1)  # [B, T, 256, H, W]
        
        # 聚合時序特徵
        aggregated_geo = self.temporal_aggregator(geo_features_tensor)  # [B, 256, H, W]
        aggregated_vmrnn = self.temporal_aggregator(vmrnn_features_tensor)  # [B, 256, H, W]
        
        # 6. 特徵融合
        fused_features = torch.cat([aggregated_geo, aggregated_vmrnn], dim=1)  # [B, 512, H, W]
        fused_features = F.relu(self.fusion_norm(self.fusion_conv(fused_features)))
        
        # 7. 預測
        prediction = self.prediction_head(fused_features)  # [B, C*output_length, H, W]
        
        # 重塑為時序輸出
        outvar_pred = prediction.view(
            batch_size, self.input_channels, self.mypara.output_length, H, W
        ).permute(0, 2, 1, 3, 4)  # [B, output_length, C, H, W]
        
        return outvar_pred
    
    def predict(self, predictor):
        """推理模式"""
        return self.forward(predictor, train=False)


# 輔助函數：創建模型的工廠函數
def create_geo_vmrnn(mypara):
    """創建 GeoVMRNN 模型的工廠函數"""
    return GeoVMRNN(mypara)
