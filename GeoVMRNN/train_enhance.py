from GeoVMRNN_enhance import *
from myconfig import mypara
import torch
from torch.utils.data import DataLoader
import numpy as np
import math
from LoadData import make_dataset2, make_testdataset
from progressive_teacher_forcing import TeacherForcingScheduler, CurriculumTeacherForcing
import mlflow
import mlflow.pytorch


class lrwarm:
    """學習率預熱調度器"""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

class ENSOLoss(nn.Module):
    """多任務損失函數：回歸 + 分類"""
    def __init__(self, classification_weight=0.1, nino_weight_path=None, device='cuda'):
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.device = device
        
        # Load Nino weights if provided
        if nino_weight_path:
            self.nino_weights = torch.load(nino_weight_path).to(device)
        else:
            # Default weights as in your original code
            ninoweight = torch.from_numpy(
                np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
                * np.log(np.arange(24) + 1)
            ).to(device)
            self.nino_weights = ninoweight
          
    
    def get_enso_labels(self, nino_indices, thresholds=(-0.5, 0.5)):
        """
        將Nino指數轉換為ENSO類別標籤
        Args:
            nino_indices: (batch, sequence_length) Nino 3.4 indices
            thresholds: (la_nina_threshold, el_nino_threshold)
        Returns:
            labels: (batch, sequence_length) ENSO labels (0=El Niño, 1=Normal, 2=La Niña)
        """
        labels = torch.ones_like(nino_indices, dtype=torch.long)  # Default to Normal
        labels[nino_indices > thresholds[1]] = 0  # El Niño
        labels[nino_indices < thresholds[0]] = 2  # La Niña
        return labels
    
    def forward(self, pred_var, true_var, pred_nino, true_nino, enso_logits=None):
        """
        計算多任務損失
        Args:
            pred_var: 預測變量 (batch, seq_len, C, H, W)
            true_var: 真實變量 (batch, seq_len, C, H, W)
            pred_nino: 預測Nino指數 (batch, seq_len)
            true_nino: 真實Nino指數 (batch, seq_len)
            enso_logits: ENSO分類預測 (batch, 3) 可選
        """
        # Variable regression loss
        var_loss = self.regression_loss(pred_var, true_var)
        
        # Nino regression loss with weighted correlation
        nino_loss = self.calculate_nino_loss(pred_nino, true_nino)
        
        total_loss = var_loss + nino_loss
        
        # Add classification loss if provided
        if enso_logits is not None:
            # Get ENSO labels from true Nino indices (use mean over sequence)
            mean_nino = true_nino.mean(dim=1)  # (batch,)
            enso_labels = self.get_enso_labels(mean_nino)
            
            classification_loss = self.classification_loss(enso_logits, enso_labels)
            total_loss += self.classification_weight * classification_loss
            
            return total_loss, var_loss, nino_loss, classification_loss, enso_labels
        
        return total_loss, var_loss, nino_loss
    
    def calculate_nino_loss(self, pred_nino, true_nino):
        """計算加權Nino損失"""
        # Ensure same sequence length
        min_len = min(pred_nino.size(1), true_nino.size(1))
        pred_nino = pred_nino[:, :min_len]
        true_nino = true_nino[:, :min_len]
        
        # Weight by time step importance
        weights = self.nino_weights[:min_len]
        
        # Weighted MSE
        mse_per_step = ((pred_nino - true_nino) ** 2).mean(dim=0)
        weighted_mse = (weights * mse_per_step).sum()
        
        return weighted_mse

    def calscore(self, y_pred, y_true):
        """計算 Nino 評分"""
        with torch.no_grad():
            pred = y_pred - y_pred.mean(dim=0, keepdim=True)
            true = y_true - y_true.mean(dim=0, keepdim=True)
            cor = (pred * true).sum(dim=0) / (
                torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0))
                + 1e-6
            )
            acc = (self.ninoweight * cor).sum()
            rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt().sum()
            sc = 2 / 3.0 * acc - rmse
        return sc.item()


class ENSOAwareTrainer:
    """ENSO感知訓練器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.criterion = ENSOLoss(device=device)

        # 添加模型參數統計
        total_params = sum(p.numel() for p in self.mymodel.parameters())
        trainable_params = sum(p.numel() for p in self.mymodel.parameters() if p.requires_grad)
        print(f"模型總參數: {total_params:,}")
        print(f"可訓練參數: {trainable_params:,}")
        print(f"使用混合激活函數配置: {self.activation_config}")

        # 設置優化器和學習率調度器
        adam = torch.optim.Adam(self.mymodel.parameters(), lr=0)
        factor = math.sqrt(mypara.d_size * mypara.warmup) * 0.0015
        self.opt = lrwarm(mypara.d_size, factor, mypara.warmup, optimizer=adam)
        
        # 設置 SST 層級
        self.sstlevel = 0
        if self.mypara.needtauxy:
            self.sstlevel = 2
        
        # 設置 Nino 指數權重
        ninoweight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
            * np.log(np.arange(24) + 1)
        ).to(mypara.device)
        self.ninoweight = ninoweight[: self.mypara.output_length]

        self.tf_scheduler = self.create_teacher_forcing_scheduler(mypara)
        self.tf_ratio_history = []

    def create_teacher_forcing_scheduler(self, mypara):
        """創建 Teacher Forcing 調度器"""
        # 從配置中獲取策略設置
        tf_strategy = getattr(mypara, 'tf_strategy', 'exponential')
        
        if tf_strategy == 'curriculum':
            # 使用課程學習調度器
            return CurriculumTeacherForcing(
                num_stages=getattr(mypara, 'tf_num_stages', 3),
                stage_steps=getattr(mypara, 'tf_stage_steps', 2000),
                strategies=[
                    {
                        'strategy': 'exponential',
                        'initial_ratio': 1.0,
                        'final_ratio': 0.7,
                        'decay_rate': 0.9998
                    },
                    {
                        'strategy': 'linear',
                        'initial_ratio': 0.7,
                        'final_ratio': 0.2,
                        'total_steps': 2000
                    },
                    {
                        'strategy': 'cosine',
                        'initial_ratio': 0.2,
                        'final_ratio': 0.0,
                        'total_steps': 2000
                    }
                ]
            )
        else:
            # 單一策略調度器的配置
            scheduler_configs = {
                'exponential': {
                    'strategy': 'exponential',
                    'initial_ratio': getattr(mypara, 'tf_initial_ratio', 1.0),
                    'final_ratio': getattr(mypara, 'tf_final_ratio', 0.0),
                    'decay_rate': getattr(mypara, 'tf_decay_rate', 0.9999)
                },
                'linear': {
                    'strategy': 'linear',
                    'initial_ratio': getattr(mypara, 'tf_initial_ratio', 1.0),
                    'final_ratio': getattr(mypara, 'tf_final_ratio', 0.0),
                    'total_steps': getattr(mypara, 'tf_total_steps', 10000)
                },
                'cosine': {
                    'strategy': 'cosine',
                    'initial_ratio': getattr(mypara, 'tf_initial_ratio', 1.0),
                    'final_ratio': getattr(mypara, 'tf_final_ratio', 0.0),
                    'total_steps': getattr(mypara, 'tf_total_steps', 10000)
                },
                'step': {
                    'strategy': 'step',
                    'initial_ratio': getattr(mypara, 'tf_initial_ratio', 1.0),
                    'final_ratio': getattr(mypara, 'tf_final_ratio', 0.0),
                    'step_size': getattr(mypara, 'tf_step_size', 1000),
                    'gamma': getattr(mypara, 'tf_gamma', 0.5)
                },
                'adaptive': {
                    'strategy': 'adaptive',
                    'initial_ratio': getattr(mypara, 'tf_initial_ratio', 1.0),
                    'final_ratio': getattr(mypara, 'tf_final_ratio', 0.0),
                    'patience': getattr(mypara, 'tf_patience', 200),
                    'threshold': getattr(mypara, 'tf_threshold', 0.01),
                    'reduction_factor': getattr(mypara, 'tf_reduction_factor', 0.8)
                }
            }
            
            config = scheduler_configs.get(tf_strategy, scheduler_configs['exponential'])
            return TeacherForcingScheduler(**config)

    
    def train_step(self, batch_data):
        """
        單步訓練，包含ENSO分類損失和殘差監督
        """
        input_var, var_true = batch_data
        
        # Forward pass - 注意這裡返回的參數順序已改變
        var_pred, residual_loss, enso_logits, attention_weights = self.model(
            input_var.float().to(self.device),
            var_true.float().to(self.device),
            train=True
        )
        
        # Extract Nino indices (same as your original logic)
        sstlevel = 2 if self.model.mypara.needtauxy else 0
        
        # True Nino indices
        SST_true = var_true[:, :, sstlevel]
        nino_true = SST_true[
            :, :,
            self.model.mypara.lat_nino_relative[0]:self.model.mypara.lat_nino_relative[1],
            self.model.mypara.lon_nino_relative[0]:self.model.mypara.lon_nino_relative[1],
        ].mean(dim=[2, 3])
        
        # Predicted Nino indices
        SST_pred = var_pred[:, :, sstlevel]
        nino_pred = SST_pred[
            :, :,
            self.model.mypara.lat_nino_relative[0]:self.model.mypara.lat_nino_relative[1],
            self.model.mypara.lon_nino_relative[0]:self.model.mypara.lon_nino_relative[1],
        ].mean(dim=[2, 3])
        
        # Calculate multi-task loss
        if enso_logits is not None:
            total_loss, var_loss, nino_loss, class_loss, enso_labels = self.criterion(
                var_pred, var_true.float().to(self.device),
                nino_pred, nino_true.float().to(self.device),
                enso_logits
            )
            
            # 加入殘差損失
            alpha = 1.0  # 可調整的殘差損失權重
            total_loss_with_residual = total_loss + alpha * residual_loss
            
            return {
                'total_loss': total_loss_with_residual,
                'var_loss': var_loss,
                'nino_loss': nino_loss,
                'classification_loss': class_loss,
                'residual_loss': residual_loss,
                'enso_labels': enso_labels,
                'attention_weights': attention_weights,
                'nino_pred': nino_pred,
                'nino_true': nino_true
            }
        else:
            total_loss, var_loss, nino_loss = self.criterion(
                var_pred, var_true.float().to(self.device),
                nino_pred, nino_true.float().to(self.device)
            )
            
            # 加入殘差損失
            alpha = 1.0
            total_loss_with_residual = total_loss + alpha * residual_loss
            
            return {
                'total_loss': total_loss_with_residual,
                'var_loss': var_loss,
                'nino_loss': nino_loss,
                'residual_loss': residual_loss,
                'nino_pred': nino_pred,
                'nino_true': nino_true
            }

def train_with_enso_classification(mypara):
    """
    使用ENSO分類增強的訓練函數
    """
    print("Creating enhanced GeoVMRNN with ENSO classification...")
    
    # Create enhanced model
    model = create_enhanced_geo_vmrnn(mypara, use_adaptive_fusion=True)
    trainer = ENSOAwareTrainer(model, device=mypara.device)
    chk_path = self.mypara.model_savepath + "ENSO_enhance.pkl"
    torch.manual_seed(self.mypara.seeds)
    print(f"模型將保存到: {chk_path}")
    
    traindataset = make_dataset2(mypara)
    evaldataset = make_testdataset(mypara, ngroup=100)
    
    dataloader_train = DataLoader(
        traindataset, batch_size=mypara.batch_size_train, shuffle=False
    )
    dataloader_eval = DataLoader(
        evaldataset, batch_size=mypara.batch_size_eval, shuffle=False
    )
    print(mypara.__dict__)
    print(f"\nloading pre-train dataset for {config_name} mixed activation model...")
    traindataset = make_dataset2(mypara)
    print(traindataset.selectregion())
    
    print(f"\nloading evaluation dataset for {config_name} mixed activation model...")
    evaldataset = make_testdataset(mypara, ngroup=100)
    print(evaldataset.selectregion())
  
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(mypara.num_epochs):
        model.train()
        epoch_losses = {
            'total': [], 'var': [], 'nino': [], 'classification': []
        }
        
        for batch_idx, batch_data in enumerate(dataloader_train):
            optimizer.zero_grad()
            
            # Enhanced training step
            results = trainer.train_step(batch_data)
            
            # Backward pass
            results['total_loss'].backward()
            optimizer.step()
            
            # Log losses
            epoch_losses['total'].append(results['total_loss'].item())
            epoch_losses['var'].append(results['var_loss'].item())
            epoch_losses['nino'].append(results['nino_loss'].item())
            
            if 'classification_loss' in results:
                epoch_losses['classification'].append(results['classification_loss'].item())
                
                # Log ENSO classification accuracy
                if batch_idx % 100 == 0:
                    enso_labels = results['enso_labels']
                    attention_weights = results['attention_weights']
                    residual_loss = results['residual_loss']
                    
                    print(f"Batch {batch_idx}")
                    print(f"  ENSO distribution: {torch.bincount(enso_labels, minlength=3)}")
                    print(f"  Attention weights mean: {attention_weights.mean(dim=0)}")
                    print(f"  Residual loss: {residual_loss.item():.4f}")
            else:
                # 沒有分類損失時也記錄殘差損失
                if batch_idx % 100 == 0:
                    residual_loss = results['residual_loss']
                    print(f"Batch {batch_idx} - Residual loss: {residual_loss.item():.4f}")
        
        # Print epoch summary
        print(f"\nEpoch {epoch}")
        print(f"  Total Loss: {np.mean(epoch_losses['total']):.4f}")
        print(f"  Var Loss: {np.mean(epoch_losses['var']):.4f}")  
        print(f"  Nino Loss: {np.mean(epoch_losses['nino']):.4f}")
        if epoch_losses['classification']:
            print(f"  Classification Loss: {np.mean(epoch_losses['classification']):.4f}")
    
    return model
