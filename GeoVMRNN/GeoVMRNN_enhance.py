import torch
import torch.nn as nn
import torch.nn.functional as F
from vmamba import VSSBlock, SS2D
from typing import Optional, Callable, Dict, Union, List
from functools import partial

class LearnedSnake(nn.Module):
    """學習性Snake激活函數"""
    def __init__(self, in_features=1, a=None):
        super().__init__()
        if a is not None:
            self.a = nn.Parameter(torch.tensor(a))
        else:
            self.a = nn.Parameter(torch.ones(1))
        self.a.requires_grad = True
        
    def forward(self, x):
        return x + torch.square(torch.sin(self.a * x)) / (torch.abs(self.a) + 1e-8)


class FixedSnake(nn.Module):
    """固定參數Snake激活函数"""
    def __init__(self, a=1.0):
        super().__init__()
        self.a = a
        
    def forward(self, x):
        return x + torch.square(torch.sin(self.a * x)) / self.a


class AdaptiveSnake(nn.Module):
    """自適應Snake激活函數"""
    def __init__(self, in_features):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.a.requires_grad = True
        
    def forward(self, x):
        return x + torch.square(torch.sin(self.a * x)) / (torch.abs(self.a) + 1e-8)


class ActivationFactory:
    """激活函數工廠類"""
    @staticmethod
    def get_activation(activation_config: Union[str, Dict], hidden_dim: int = None):
        """
        根據配置創建激活函數
        
        Args:
            activation_config: 字符串或字典配置
            hidden_dim: 隱藏維度（某些激活函數需要）
            
        Returns:
            nn.Module: 激活函數
        """
        if isinstance(activation_config, str):
            activation_type = activation_config
            params = {}
        else:
            activation_type = activation_config.get('type', 'relu')
            params = activation_config.get('params', {})
        
        if activation_type == 'relu':
            return nn.ReLU(inplace=params.get('inplace', True))
        elif activation_type == 'gelu':
            return nn.GELU()
        elif activation_type == 'silu':
            return nn.SiLU(inplace=params.get('inplace', True))
        elif activation_type == 'snake_learned':
            return LearnedSnake(in_features=params.get('in_features', hidden_dim))
        elif activation_type == 'snake_fixed':
            return FixedSnake(a=params.get('a', 1.0))
        elif activation_type == 'snake_adaptive':
            return AdaptiveSnake(in_features=params.get('in_features', hidden_dim))
        else:
            return nn.ReLU(inplace=True)
        
class VSB(VSSBlock):
    def __init__(
        self,
        hidden_dim: int = 0,
        input_resolution: tuple = None,  # None，強制傳入
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs
    ):
        # 如果沒有傳入 input_resolution，使用默認值
        if input_resolution is None:
            input_resolution = (224, 224)
            
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
            VSB(hidden_dim=hidden_dim, 
                input_resolution=input_resolution,  # 確保傳入正確的分辨率
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
    def __init__(self, in_channels, out_channels, activation_config, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = ActivationFactory.get_activation(activation_config, out_channels)
        # self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class SpatialAttention(nn.Module):
    """空間注意力機制"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class ENSOClassifier(nn.Module):
    """ENSO現象分類器"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # 3 classes: El Niño, Normal, La Niña
        )
        
    def forward(self, x):
        return self.classifier(x)


class AdaptiveFusionModule(nn.Module):
    """自適應融合模塊，根據ENSO狀態動態選擇激活函數"""
    def __init__(self, input_channels, hidden_dim=512):
        super().__init__()
        
        # Different activation branches
        self.relu_branch = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.snake_branch = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            LearnedSnake()
        )
        
        # Classification head for determining ENSO state
        self.enso_classifier = ENSOClassifier(input_channels, hidden_dim=128)
        
        # Attention weights for fusion
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 activation branches
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Get features from different activation branches
        relu_feat = self.relu_branch(x)
        snake_feat = self.snake_branch(x)
        
        # Get ENSO classification
        enso_logits = self.enso_classifier(x)
        enso_probs = F.softmax(enso_logits, dim=1)
        
        # Get attention weights
        attention_weights = self.attention(x)
        
        # Adaptive fusion based on ENSO classification
        # El Niño (class 0) -> prefer ReLU
        # Normal (class 1) -> balanced
        # La Niña (class 2) -> prefer Snake
        fused_features = (
            attention_weights[:, 0:1].unsqueeze(-1).unsqueeze(-1) * relu_feat +
            attention_weights[:, 1:2].unsqueeze(-1).unsqueeze(-1) * snake_feat +
            attention_weights[:, 2:3].unsqueeze(-1).unsqueeze(-1) * snake_feat
        )
        
        return fused_features, enso_logits, attention_weights
        
class SupervisedPredictionHead(nn.Module):
    """支持監督式學習的預測頭"""
    def __init__(self, input_dim, cube_dim, activation_config, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.cube_dim = cube_dim
        self.num_layers = num_layers
        
        layers = []
        hidden_dim = input_dim // 2
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1))
            elif i == num_layers - 1:
                layers.append(nn.Conv2d(hidden_dim, cube_dim, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(hidden_dim))
                activation = ActivationFactory.get_activation(activation_config, hidden_dim)
                layers.append(activation)
        
        self.prediction_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.prediction_layers(x)


class EnhancedGeoVMRNN_Supervised(nn.Module):
    """增強的GeoVMRNN模型，支持ENSO分類和自適應激活函數"""
    def __init__(self, mypara, activation_config=None, use_adaptive_fusion=True):
        super().__init__()
        self.mypara = mypara
        self.device = mypara.device
        self.use_adaptive_fusion = use_adaptive_fusion

        # 默認激活函數配置
        if activation_config is None:
            activation_config = 'relu'
        self.activation_config = activation_config

        # 使用與 Geoformer 相同的 patch 參數
        if hasattr(mypara, 'patch_size'):
            self.patch_size = mypara.patch_size
        else:
            self.patch_size = (4, 4)
            
        # 計算地理尺寸
        if hasattr(mypara, 'lat_range') and hasattr(mypara, 'lon_range'):
            # 根據地理範圍計算实际尺寸
            lat_span = mypara.lat_range[1] - mypara.lat_range[0]
            lon_span = mypara.lon_range[1] - mypara.lon_range[0]
            
            # 如果有分辨率信息，使用分辨率計算
            if hasattr(mypara, 'resolution'):
                self.img_height = int(lat_span / mypara.resolution)
                self.img_width = int(lon_span / mypara.resolution)
            else:
                # 否則直接使用度數作為像素數（可能需要調整）
                self.img_height = int(lat_span)
                self.img_width = int(lon_span)
                
        elif hasattr(mypara, 'H0') and hasattr(mypara, 'W0'):
            # 如果直接給出了patch后的尺寸
            self.H0 = mypara.H0
            self.W0 = mypara.W0
            self.img_height = self.H0 * self.patch_size[0]
            self.img_width = self.W0 * self.patch_size[1]
        else:
            # 默認值
            self.img_height = 224
            self.img_width = 224
            
        # 計算patch後的尺寸
        self.H0 = self.img_height // self.patch_size[0]
        self.W0 = self.img_width // self.patch_size[1]
        self.emb_spatial_size = self.H0 * self.W0
        
        # 地理輸入分辨率（patch後的尺寸）
        self.geo_input_resolution = (self.H0, self.W0)
            
        # 計算cube_dim（與Geoformer保持一致）
        if self.mypara.needtauxy:
            self.cube_dim = mypara.input_channal + 2
            self.input_channels = mypara.input_channal + 2
        else:
            self.cube_dim = mypara.input_channal
            self.input_channels = mypara.input_channal

        # 處理激活函數配置
        cnn_config, fusion_config, pred_config = self._parse_activation_config()

        # 編碼器：地理空間特徵提取
        self.geo_cnn = nn.Sequential(
            GeoCNN(self.input_channels, 64, activation_config=cnn_config),
            GeoCNN(64, 128, activation_config=cnn_config),
            GeoCNN(128, 256, activation_config=cnn_config),
            SpatialAttention(256)
        )
        
        # 嵌入維度
        self.embed_dim = 256
        
        # 編碼器：VMRNN Cell（使用地理分辨率）
        self.encoder_vmrnn_cell = VMRNNCell(
            hidden_dim=self.embed_dim,
            input_resolution=self.geo_input_resolution,  # 使用地理分辨率
            depth=2,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            d_state=16
        )
        
        # 解碼器：VMRNN Cell（使用地理分辨率）
        self.decoder_vmrnn_cell = VMRNNCell(
            hidden_dim=self.embed_dim,
            input_resolution=self.geo_input_resolution,  # 使用地理分辨率
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
        # Replace fusion layer with adaptive fusion if enabled
        if self.use_adaptive_fusion:
            self.adaptive_fusion = AdaptiveFusionModule(
                input_channels=256 + 256,  # geo_feat + vmrnn_feat
                hidden_dim=512
            )
        else:
            # Keep original fusion
            self.fusion_conv = nn.Conv2d(256 + 256, 512, kernel_size=1)
            self.fusion_norm = nn.BatchNorm2d(512)
            self.fusion_activation = ActivationFactory.get_activation(
                activation_config.get('fusion', 'relu') if isinstance(activation_config, dict) else 'relu', 
                512
            )

        
        # 監督式預測頭（每次只預測一個時間步）
        self.prediction_head = SupervisedPredictionHead(
            input_dim=512,
            cube_dim=self.cube_dim,
            num_layers=3,
            activation_config=pred_config
        )
      
        # 初始化誤差預測
        self.residual_head = SupervisedPredictionHead(
            input_dim=512,  # 或與 fusion_conv 輸出一致
            cube_dim=self.cube_dim,
            num_layers=2,
            activation_config=pred_config
        )
        # self.error_correction_ann = ErrorCorrectionANN(input_dim=self.cube_dim)
        
        # 打印調試信息
        print(f"地理尺寸: {self.img_height} x {self.img_width}")
        print(f"Patch后尺寸: {self.H0} x {self.W0}")
        print(f"使用的input_resolution: {self.geo_input_resolution}")
        print(f"使用混合激活函數配置: {self.activation_config}")
        self._print_activation_summary()

    def _parse_activation_config(self):
        """解析激活函數配置"""
        if isinstance(self.activation_config, str):
            # 全部使用相同激活函數
            return (self.activation_config,) * 3
        elif isinstance(self.activation_config, dict):
            # 字典形式配置
            cnn_config = self.activation_config.get('cnn', 'relu')
            fusion_config = self.activation_config.get('fusion', 'relu')
            pred_config = self.activation_config.get('prediction', 'relu')
            return cnn_config, fusion_config, pred_config
        else:
            return ('relu',) * 3

    def _print_activation_summary(self):
        """打印激活函數使用摘要"""
        cnn_config, fusion_config, pred_config = self._parse_activation_config()
        
        print("激活函數配置摘要:")
        print(f"  CNN特徵提取層: {cnn_config}")
        print(f"  特徵融合層: {fusion_config}")
        print(f"  預測頭: {pred_config}")
        
    def encode(self, predictor):
        """編碼器：處理歷史數據"""
        batch_size, seq_len, C, H, W = predictor.shape
        # print(f"Encode - Input shape: {predictor.shape}")
        
        # 验证输入尺寸是否匹配
        assert H == self.img_height and W == self.img_width, \
            f"输入尺寸 {H}x{W} 不匹配预期的地理尺寸 {self.img_height}x{self.img_width}"
        
        # 初始化編碼器隱藏狀態
        encoder_hidden = None
        
        # 編碼所有歷史時間步
        for t in range(seq_len):
            # 地理空間特徵提取
            geo_feat = self.geo_cnn(predictor[:, t])
            # print(f"Encode - After CNN shape at step {t}: {geo_feat.shape}")
            
            # 轉換為patch序列
            patch_feat = self.patch_project(geo_feat)
            B, C_embed, H_patch, W_patch = patch_feat.shape
            # print(f"Encode - After patch projection shape at step {t}: {patch_feat.shape}")
            
            # 验证patch尺寸
            assert H_patch == self.H0 and W_patch == self.W0, \
                f"Patch尺寸 {H_patch}x{W_patch} 不匹配预期的 {self.H0}x{self.W0}"
            
            patch_embed_feat = patch_feat.view(B, C_embed, H_patch * W_patch).permute(0, 2, 1)
            # print(f"Encode - After reshape and permute at step {t}: {patch_embed_feat.shape}")
            
            # 編碼器VMRNN處理
            encoded_feat, encoder_hidden = self.encoder_vmrnn_cell(patch_embed_feat, encoder_hidden)
            # print(f"Encode - After VMRNN at step {t}: {encoded_feat.shape}")
        
        # print(f"Encode - Final output shape: {encoded_feat.shape}")
        return encoded_feat, encoder_hidden

    def enhanced_decode_step_with_residual(self, current_input, encoder_output, decoder_hidden):
        """
        增強的解碼步驟，包含自適應融合和殘差校正
        """
        # Original decode steps (same as your decode_step method)
        geo_feat = self.geo_cnn(current_input)
        patch_feat = self.patch_project(geo_feat)
        B, C_embed, H_patch, W_patch = patch_feat.shape
        patch_embed_feat = patch_feat.view(B, C_embed, H_patch * W_patch).permute(0, 2, 1)
        
        decoded_feat, new_decoder_hidden = self.decoder_vmrnn_cell(patch_embed_feat, decoder_hidden)
        decoded_img = decoded_feat.permute(0, 2, 1).view(B, self.embed_dim, H_patch, W_patch)
        vmrnn_feat = self.patch_to_img(decoded_img)
        
        encoder_img = encoder_output.permute(0, 2, 1).view(B, self.embed_dim, H_patch, W_patch)
        encoder_feat = self.patch_to_img(encoder_img)
        
        # Concatenate features for fusion
        combined_features = torch.cat([geo_feat, vmrnn_feat], dim=1)
        
        # Apply adaptive or standard fusion
        if self.use_adaptive_fusion:
            fused_features, enso_logits, attention_weights = self.adaptive_fusion(combined_features)
        else:
            fused_features = self.fusion_activation(
                self.fusion_norm(self.fusion_conv(combined_features))
            )
            enso_logits = None
            attention_weights = None
        
        # 粗預測頭 (coarse prediction)
        coarse_prediction = self.prediction_head(fused_features)
        
        # 殘差預測頭 (residual prediction)  
        residual_prediction = self.residual_head(fused_features)
        
        # 最終預測 = 粗預測 + 殘差校正
        final_prediction = coarse_prediction + residual_prediction
        
        # Reshape to original dimensions
        H, W = current_input.shape[-2:]
        coarse_prediction = coarse_prediction.view(B, self.cube_dim, H, W)
        residual_prediction = residual_prediction.view(B, self.cube_dim, H, W)
        final_prediction = final_prediction.view(B, self.cube_dim, H, W)
        
        return final_prediction, coarse_prediction, residual_prediction, enso_logits, attention_weights, new_decoder_hidden

        def forward(self, predictor, predictand=None, train=True, sv_ratio=0):
        """
        前向傳播，支持ENSO分類並保留殘差校正結構
        """
        batch_size, seq_len, C, H, W = predictor.shape
        
        # 1. Encoding phase
        encoder_output, encoder_hidden = self.encode(predictor)
        
        # 2. Decoding phase with enhanced fusion and residual correction
        if train:
            assert predictand is not None, "在訓練模式下必須提供 predictand"
            
            decoder_hidden = encoder_hidden
            outputs = []
            classification_losses = []
            attention_weights_list = []
            mse = nn.MSELoss()
            residual_losses = []
            
            current_input = predictor[:, -1]
            
            for t in range(self.mypara.output_length):
                # Enhanced decode step with residual correction
                next_step, coarse_pred, residual_pred, enso_logits, attention_weights, decoder_hidden = \
                    self.enhanced_decode_step_with_residual(current_input, encoder_output, decoder_hidden)
                
                outputs.append(next_step)
                
                # 殘差監督：(真實值 - 粗預測) 作為殘差目標
                residual_target = predictand[:, t] - coarse_pred
                loss_r = mse(residual_pred, residual_target)
                residual_losses.append(loss_r)
                
                # 收集分類和注意力權重信息
                if enso_logits is not None:
                    classification_losses.append(enso_logits)
                    attention_weights_list.append(attention_weights)
                
                # Teacher forcing
                if t < self.mypara.output_length - 1:
                    current_input = predictand[:, t]
            
            outvar_pred = torch.stack(outputs, dim=1)
            
            # 應用監督比例（Teacher Forcing with mixed input）
            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio * torch.ones(batch_size, self.mypara.output_length - 1, 1, 1, 1)
                ).to(self.device)
                
                # 混合真實值和預測值
                mixed_predictand = (
                    supervise_mask * predictand[:, :-1] + 
                    (1 - supervise_mask) * outvar_pred[:, :-1]
                )
                
                # 重新進行預測
                decoder_hidden = encoder_hidden
                outputs = []
                residual_losses = []
                classification_losses = []
                attention_weights_list = []
                current_input = predictor[:, -1]
                
                for t in range(self.mypara.output_length):
                    next_step, coarse_pred, residual_pred, enso_logits, attention_weights, decoder_hidden = \
                        self.enhanced_decode_step_with_residual(current_input, encoder_output, decoder_hidden)
                    
                    outputs.append(next_step)
                    
                    # 殘差監督
                    residual_target = predictand[:, t] - coarse_pred
                    loss_r = mse(residual_pred, residual_target)
                    residual_losses.append(loss_r)
                    
                    # 收集分類信息
                    if enso_logits is not None:
                        classification_losses.append(enso_logits)
                        attention_weights_list.append(attention_weights)
                    
                    # 使用混合的輸入
                    if t < self.mypara.output_length - 1:
                        current_input = mixed_predictand[:, t]
                
                outvar_pred = torch.stack(outputs, dim=1)
            
            # 計算平均殘差損失
            mean_residual_loss = torch.stack(residual_losses).mean()
            
            # 處理分類結果
            if classification_losses:
                avg_enso_logits = torch.stack(classification_losses).mean(dim=0)
                avg_attention_weights = torch.stack(attention_weights_list).mean(dim=0)
                return outvar_pred, mean_residual_loss, avg_enso_logits, avg_attention_weights
            else:
                return outvar_pred, mean_residual_loss, None, None
        
        else:
            # 推理模式：自回歸生成
            decoder_hidden = encoder_hidden
            outputs = []
            residual_losses = []
            current_input = predictor[:, -1]
            
            for t in range(self.mypara.output_length):
                next_step, coarse_pred, residual_pred, enso_logits, attention_weights, decoder_hidden = \
                    self.enhanced_decode_step_with_residual(current_input, encoder_output, decoder_hidden)
                
                outputs.append(next_step)
                current_input = next_step  # 使用預測結果作為下一步輸入
                
                # 推理模式下設置殘差損失為零（沒有真實值可比較）
                if predictand is not None:
                    mse = nn.MSELoss()
                    residual_target = predictand[:, t] - coarse_pred
                    loss_r = mse(residual_pred, residual_target)
                    residual_losses.append(loss_r)
                else:
                    residual_losses.append(torch.tensor(0.0, device=self.device))
            
            outvar_pred = torch.stack(outputs, dim=1)
            
            # 返回平均殘差損失
            mean_residual_loss = torch.stack(residual_losses).mean() if residual_losses else torch.tensor(0.0, device=self.device)
            
            return outvar_pred, mean_residual_loss, None, None
    
    def predict(self, predictor):
        """推理模式"""
        return self.forward(predictor, train=False)

# 添加維度檢查
def check_dimensions(self, x, stage):
    print(f"{stage}: {x.shape}")
    return x

# 工廠函數
def create_supervised_geo_vmrnn(mypara, activation_config=None):
    """創建支持監督式學習的 GeoVMRNN 模型"""
    return GeoVMRNN_Supervised(mypara, activation_config=activation_config)
