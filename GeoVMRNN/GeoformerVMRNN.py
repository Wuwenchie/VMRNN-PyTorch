import torch
import torch.nn as nn
import torch.nn.functional as F
from my_tools import make_embedding, unfold_func, miniEncoder, miniDecoder, fold_func


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
        # x: [B, C, H, W]
        attention = self.sigmoid(self.conv(x))
        return x * attention


class VMRNNCell(nn.Module):
    """VMRNN單元 - 用於處理時間序列"""
    def __init__(self, input_size, hidden_size, kernel_size=3, padding=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = padding
        
        # 輸入到隱藏狀態的卷積
        self.conv_ih = nn.Conv2d(input_size, 4 * hidden_size, kernel_size, padding=padding)
        # 隱藏狀態到隱藏狀態的卷積
        self.conv_hh = nn.Conv2d(hidden_size, 4 * hidden_size, kernel_size, padding=padding)
        
    def forward(self, x, h_prev, c_prev):
        # x: [B, C, H, W]
        # h_prev, c_prev: [B, hidden_size, H, W]
        
        gates_ih = self.conv_ih(x)
        gates_hh = self.conv_hh(h_prev)
        gates = gates_ih + gates_hh
        
        # 分割為四個門
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
        
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        
        c_new = f_gate * c_prev + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)
        
        return h_new, c_new


class GeoformerVMRNN(nn.Module):
    """Geoformer與VMRNN的融合模型"""
    def __init__(self, mypara):
        super().__init__()
        self.mypara = mypara
        self.device = mypara.device
        d_size = mypara.d_size
        
        # 計算輸入維度
        if self.mypara.needtauxy:
            self.input_channels = mypara.input_channal + 2
        else:
            self.input_channels = mypara.input_channal
            
        self.cube_dim = self.input_channels * mypara.patch_size[0] * mypara.patch_size[1]
        
        # 地理空間特徵提取
        self.geo_cnn = nn.Sequential(
            GeoCNN(self.input_channels, 64),
            GeoCNN(64, 128),
            GeoCNN(128, 256),
            SpatialAttention(256)
        )
        
        # VMRNN層
        self.vmrnn_hidden_size = 128
        self.vmrnn = VMRNNCell(256, self.vmrnn_hidden_size)
        
        # Transformer編碼器
        self.predictor_emb = make_embedding(
            cube_dim=self.cube_dim,
            d_size=d_size,
            emb_spatial_size=mypara.emb_spatial_size,
            max_len=mypara.input_length,
            device=self.device,
        )
        
        enc_layer = miniEncoder(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout
        )
        self.encoder = multi_enc_layer(
            enc_layer=enc_layer, num_layers=mypara.num_encoder_layers
        )
        
        # 解碼器
        self.predictand_emb = make_embedding(
            cube_dim=self.cube_dim,
            d_size=d_size,
            emb_spatial_size=mypara.emb_spatial_size,
            max_len=mypara.output_length,
            device=self.device,
        )
        
        dec_layer = miniDecoder(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout
        )
        self.decoder = multi_dec_layer(
            dec_layer=dec_layer, num_layers=mypara.num_decoder_layers
        )
        
        # 特徵融合層
        self.fusion_conv = nn.Conv2d(self.vmrnn_hidden_size + 256, 256, kernel_size=1)
        
        # 輸出層
        self.linear_output = nn.Linear(d_size, self.cube_dim)
        
    def forward(self, predictor, predictand=None, in_mask=None, enout_mask=None, train=True, sv_ratio=0):
        """
        Args:
            predictor: (batch, lb, C, H, W)
            predictand: (batch, pre_len, C, H, W)
        Returns:
            outvar_pred: (batch, pre_len, C, H, W)
        """
        # 1. 地理空間特徵提取 + VMRNN處理
        batch_size, seq_len, C, H, W = predictor.shape
        
        # 初始化VMRNN狀態
        h_state = torch.zeros(batch_size, self.vmrnn_hidden_size, H, W).to(self.device)
        c_state = torch.zeros(batch_size, self.vmrnn_hidden_size, H, W).to(self.device)
        
        vmrnn_features = []
        geo_features = []
        
        for t in range(seq_len):
            # 地理空間特徵提取
            geo_feat = self.geo_cnn(predictor[:, t])  # [B, 256, H, W]
            geo_features.append(geo_feat)
            
            # VMRNN處理
            h_state, c_state = self.vmrnn(geo_feat, h_state, c_state)
            vmrnn_features.append(h_state)
        
        # 2. 特徵融合
        fused_features = []
        for t in range(seq_len):
            # 融合地理特徵和VMRNN特徵
            concat_feat = torch.cat([geo_features[t], vmrnn_features[t]], dim=1)
            fused_feat = self.fusion_conv(concat_feat)  # [B, 256, H, W]
            fused_features.append(fused_feat)
        
        # 3. Transformer編碼
        en_out = self.encode_with_fusion(predictor, fused_features, in_mask)
        
        # 4. 解碼
        if train:
            # 訓練時的teacher forcing
            if predictand is None:
                raise ValueError("predictand is required during training")
                
            with torch.no_grad():
                connect_inout = torch.cat([predictor[:, -1:], predictand[:, :-1]], dim=1)
                out_mask = self.make_mask_matrix(connect_inout.size(1))
                outvar_pred = self.decode(connect_inout, en_out, out_mask, enout_mask)
            
            # 監督式學習
            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio * torch.ones(predictand.size(0), predictand.size(1) - 1, 1, 1, 1)
                ).to(self.device)
            else:
                supervise_mask = 0
                
            predictand = (
                supervise_mask * predictand[:, :-1] + 
                (1 - supervise_mask) * outvar_pred[:, :-1]
            )
            predictand = torch.cat([predictor[:, -1:], predictand], dim=1)
            
            # 最終預測
            outvar_pred = self.decode(predictand, en_out, out_mask, enout_mask)
        else:
            # 推理時的自迴歸解碼
            predictand = predictor[:, -1:]
            for t in range(self.mypara.output_length):
                out_mask = self.make_mask_matrix(predictand.size(1))
                outvar_pred = self.decode(predictand, en_out, out_mask, enout_mask)
                predictand = torch.cat([predictand, outvar_pred[:, -1:]], dim=1)
        
        return outvar_pred
    
    def encode_with_fusion(self, predictor, fused_features, in_mask):
        """融合特徵的編碼"""
        lb = predictor.size(1)
        
        # 使用原始的patch embedding
        predictor_patches = unfold_func(predictor, self.mypara.patch_size)
        predictor_patches = predictor_patches.reshape(
            predictor_patches.size(0), lb, self.cube_dim, -1
        ).permute(0, 3, 1, 2)
        
        # Transformer編碼
        predictor_emb = self.predictor_emb(predictor_patches)
        en_out = self.encoder(predictor_emb, in_mask)
        
        return en_out
    
    def decode(self, predictand, en_out, out_mask, enout_mask):
        """解碼器"""
        H, W = predictand.size()[-2:]
        T = predictand.size(1)
        
        predictand_patches = unfold_func(predictand, self.mypara.patch_size)
        predictand_patches = predictand_patches.reshape(
            predictand_patches.size(0), T, self.cube_dim, -1
        ).permute(0, 3, 1, 2)
        
        predictand_emb = self.predictand_emb(predictand_patches)
        output = self.decoder(predictand_emb, en_out, out_mask, enout_mask)
        output = self.linear_output(output).permute(0, 2, 3, 1)
        
        output = output.reshape(
            predictand_patches.size(0), T, self.cube_dim,
            H // self.mypara.patch_size[0], W // self.mypara.patch_size[1]
        )
        
        output = fold_func(
            output, output_size=(H, W), kernel_size=self.mypara.patch_size
        )
        
        return output
    
    def make_mask_matrix(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 0).T
        return mask.to(self.device)


class multi_enc_layer(nn.Module):
    def __init__(self, enc_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([enc_layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class multi_dec_layer(nn.Module):
    def __init__(self, dec_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([dec_layer for _ in range(num_layers)])

    def forward(self, x, en_out, out_mask, enout_mask):
        for layer in self.layers:
            x = layer(x, en_out, out_mask, enout_mask)
        return x
