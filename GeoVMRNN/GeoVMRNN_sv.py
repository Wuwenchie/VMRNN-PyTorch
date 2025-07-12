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


class SupervisedPredictionHead(nn.Module):
    """支持監督式學習的預測頭"""
    def __init__(self, input_dim, cube_dim, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.cube_dim = cube_dim
        self.num_layers = num_layers
        
        # 單步預測網路（每次只預測一個時間步）
        layers = []
        hidden_dim = input_dim // 2
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1))
            elif i == num_layers - 1:
                layers.append(nn.Conv2d(hidden_dim, cube_dim, kernel_size=3, padding=1))  # 只預測一個時間步
            else:
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
        
        self.prediction_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.prediction_layers(x)


class GeoVMRNN_Supervised(nn.Module):
    """支持監督式學習的 GeoVMRNN 模型"""
    def __init__(self, mypara):
        super().__init__()
        self.mypara = mypara
        self.device = mypara.device
        
        # 使用與 Geoformer 相同的 patch 參數
        if hasattr(mypara, 'patch_size'):
            self.patch_size = mypara.patch_size
        else:
            self.patch_size = (4, 4)
            
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
                self.img_height = 224
                self.img_width = 224
            
            self.H0 = self.img_height // self.patch_size[0]
            self.W0 = self.img_width // self.patch_size[1]
            self.emb_spatial_size = self.H0 * self.W0
        
        # 編碼器：地理空間特徵提取
        self.geo_cnn = nn.Sequential(
            GeoCNN(self.input_channels, 64),
            GeoCNN(64, 128),
            GeoCNN(128, 256),
            SpatialAttention(256)
        )
        
        # 嵌入維度
        self.embed_dim = 256
        
        # 編碼器：VMRNN Cell
        self.encoder_vmrnn_cell = VMRNNCell(
            hidden_dim=self.embed_dim,
            input_resolution=(self.H0, self.W0),
            depth=2,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            d_state=16
        )
        
        # 解碼器：VMRNN Cell（用於自回歸生成）
        self.decoder_vmrnn_cell = VMRNNCell(
            hidden_dim=self.embed_dim,
            input_resolution=(self.H0, self.W0),
            depth=2,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            d_state=16
        )
        
        # patch嵌入層
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
        
        # 特徵融合層
        self.fusion_conv = nn.Conv2d(256 + 256, 512, kernel_size=1)
        self.fusion_norm = nn.BatchNorm2d(512)
        
        # 監督式預測頭（每次只預測一個時間步）
        self.prediction_head = SupervisedPredictionHead(
            input_dim=512,
            cube_dim=self.cube_dim,
            num_layers=3
        )
        
    def encode(self, predictor):
        """編碼器：處理歷史數據"""
        batch_size, seq_len, C, H, W = predictor.shape
        
        # 初始化編碼器隱藏狀態
        encoder_hidden = None
        
        # 編碼所有歷史時間步
        for t in range(seq_len):
            # 地理空間特徵提取
            geo_feat = self.geo_cnn(predictor[:, t])
            
            # 轉換為patch序列
            patch_feat = self.patch_project(geo_feat)
            B, C_embed, H_patch, W_patch = patch_feat.shape
            patch_embed_feat = patch_feat.view(B, C_embed, H_patch * W_patch).permute(0, 2, 1)
            
            # 編碼器VMRNN處理
            encoded_feat, encoder_hidden = self.encoder_vmrnn_cell(patch_embed_feat, encoder_hidden)
        
        return encoded_feat, encoder_hidden
    
    def decode_step(self, current_input, encoder_output, decoder_hidden):
        """解碼器：單步預測"""
        # 地理空間特徵提取
        geo_feat = self.geo_cnn(current_input)
        
        # 轉換為patch序列
        patch_feat = self.patch_project(geo_feat)
        B, C_embed, H_patch, W_patch = patch_feat.shape
        patch_embed_feat = patch_feat.view(B, C_embed, H_patch * W_patch).permute(0, 2, 1)
        
        # 解碼器VMRNN處理
        decoded_feat, new_decoder_hidden = self.decoder_vmrnn_cell(patch_embed_feat, decoder_hidden)
        
        # 轉換回圖像格式
        decoded_img = decoded_feat.permute(0, 2, 1).view(B, self.embed_dim, H_patch, W_patch)
        vmrnn_feat = self.patch_to_img(decoded_img)
        
        # 特徵融合
        encoder_img = encoder_output.permute(0, 2, 1).view(B, self.embed_dim, H_patch, W_patch)
        encoder_feat = self.patch_to_img(encoder_img)
        
        fused_features = torch.cat([geo_feat, vmrnn_feat], dim=1)
        fused_features = F.relu(self.fusion_norm(self.fusion_conv(fused_features)))
        
        # 預測下一時間步
        prediction = self.prediction_head(fused_features)
        
        # 恢復到原始尺寸
        H, W = current_input.shape[-2:]
        prediction = prediction.view(B, self.cube_dim, self.H0, self.W0)
        next_step = fold_func(prediction, output_size=(H, W), kernel_size=self.patch_size)
        
        return next_step, new_decoder_hidden
    
    def forward(self, predictor, predictand=None, train=True, sv_ratio=0):
        """
        Args:
            predictor: (batch, input_length, C, H, W)
            predictand: (batch, output_length, C, H, W) - 訓練時使用
            train: 是否為訓練模式
            sv_ratio: 監督比例（與Geoformer一致）
        Returns:
            outvar_pred: (batch, output_length, C, H, W)
        """
        batch_size, seq_len, C, H, W = predictor.shape
        
        # 1. 編碼階段
        encoder_output, encoder_hidden = self.encode(predictor)
        
        # 2. 解碼階段
        if train:
            # 訓練模式：使用teacher forcing
            assert predictand is not None
            
            # 創建解碼輸入序列（與Geoformer相同的方式）
            connect_inout = torch.cat([predictor[:, -1:], predictand[:, :-1]], dim=1)
            
            # 監督式訓練
            decoder_hidden = encoder_hidden
            outputs = []
            
            for t in range(self.mypara.output_length):
                # 使用當前輸入預測下一步
                current_input = connect_inout[:, t]
                next_step, decoder_hidden = self.decode_step(
                    current_input, encoder_output, decoder_hidden
                )
                outputs.append(next_step)
            
            outvar_pred = torch.stack(outputs, dim=1)
            
            # 應用監督比例（與Geoformer相同）
            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio * torch.ones(batch_size, self.mypara.output_length - 1, 1, 1, 1)
                ).to(self.device)
                
                # 混合真實值和預測值
                mixed_predictand = (
                    supervise_mask * predictand[:, :-1] + 
                    (1 - supervise_mask) * outvar_pred[:, :-1]
                )
                
                # 重新預測
                connect_inout = torch.cat([predictor[:, -1:], mixed_predictand], dim=1)
                decoder_hidden = encoder_hidden
                outputs = []
                
                for t in range(self.mypara.output_length):
                    current_input = connect_inout[:, t]
                    next_step, decoder_hidden = self.decode_step(
                        current_input, encoder_output, decoder_hidden
                    )
                    outputs.append(next_step)
                
                outvar_pred = torch.stack(outputs, dim=1)
            
        else:
            # 推理模式：自回歸生成
            decoder_hidden = encoder_hidden
            outputs = []
            
            # 使用編碼器最後一個時間步作為解碼器初始輸入
            current_input = predictor[:, -1]
            
            for t in range(self.mypara.output_length):
                next_step, decoder_hidden = self.decode_step(
                    current_input, encoder_output, decoder_hidden
                )
                outputs.append(next_step)
                current_input = next_step  # 使用預測結果作為下一步輸入
            
            outvar_pred = torch.stack(outputs, dim=1)
        
        return outvar_pred
    
    def predict(self, predictor):
        """推理模式"""
        return self.forward(predictor, train=False)


# 工廠函數
def create_supervised_geo_vmrnn(mypara):
    """創建支持監督式學習的 GeoVMRNN 模型"""
    return GeoVMRNN_Supervised(mypara)
