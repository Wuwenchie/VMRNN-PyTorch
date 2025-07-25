from GeoVMRNN_ablation import create_supervised_geo_vmrnn, get_ablation_configs
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


class GeoVMRNNMixedActivationTrainer:
    """GeoVMRNN 混合激活函數模型訓練器"""
    def __init__(self, mypara, activation_config=None):
        assert mypara.input_channal == mypara.output_channal
        
        # 添加數據維度驗證
        print("正在驗證數據維度...")
        
        self.mypara = mypara
        self.device = mypara.device
        # self.activation_config = activation_config or 'relu'
        self.activation_config = activation_config
                
        # 創建混合激活函數模型
        self.mymodel = create_supervised_geo_vmrnn(
            mypara, 
            activation_config=self.activation_config
        ).to(mypara.device)
        
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
        # 从配置中获取策略设置
        tf_strategy = getattr(mypara, 'tf_strategy', 'exponential')
        
        if tf_strategy == 'curriculum':
            # 使用课程学习调度器
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
            # 单一策略调度器的配置
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

    def get_model_name(self):
        """根據激活函數配置生成模型名稱"""
        if isinstance(self.activation_config, str):
            return f"GeoVMRNN_Mixed_{self.activation_config}.pkl"
        elif isinstance(self.activation_config, dict):
            # 創建簡短的配置描述
            parts = []
            for key, value in self.activation_config.items():
                if isinstance(value, str):
                    parts.append(f"{key}_{value}")
                elif isinstance(value, dict):
                    # 對於嵌套字典，只取主要信息
                    main_info = value.get('vsb', value.get('type', 'mixed'))
                    if isinstance(main_info, list):
                        main_info = '_'.join(main_info[:2])  # 只取前兩個
                    parts.append(f"{key}_{main_info}")
            config_str = '_'.join(parts[:3])  # 限制長度
            return f"GeoVMRNN_Mixed_{config_str}.pkl"
        else:
            return "GeoVMRNN_Mixed_custom.pkl"
        
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

    def loss_var(self, y_pred, y_true, residual_losses=None, alpha=1.0):
        """計算變量損失"""
        # Ensure y_pred and y_true have the same sequence length
        min_len = min(y_pred.size(1), y_true.size(1))
        y_pred = y_pred[:, :min_len]
        y_true = y_true[:, :min_len]
        
        # Calculate RMSE over spatial dimensions first
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])  # Average over height and width
        
        # Average over batch dimension
        rmse = rmse.mean(dim=0)  # Average over batch
        
        # Sum over remaining dimensions (sequence and channels)
        rmse = rmse.sum()
        
        # 處理 residual_losses
        if residual_losses is not None:
            # 如果 residual_losses 是 tensor，直接使用
            if isinstance(residual_losses, torch.Tensor):
                residual_term = alpha * residual_losses
            else:
                # 如果是其他類型（如 float），轉換為 tensor
                residual_term = alpha * torch.tensor(residual_losses, device=rmse.device)
        else:
            # 如果沒有提供 residual_losses，設為 0
            residual_term = torch.tensor(0.0, device=rmse.device)
        
        total_loss = rmse + residual_term
        return total_loss

    def loss_nino(self, y_pred, y_true):
        """計算 Nino 損失"""
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()

    def combine_loss(self, loss1, loss2):
        """組合損失函數"""
        combine_loss = loss1 + loss2
        return combine_loss

    def model_pred(self, dataloader):
        """模型預測和評估"""
        self.mymodel.eval()
        nino_pred = []
        var_pred = []
        nino_true = []
        var_true = []
        
        with torch.no_grad():
            for input_var, var_true1 in dataloader:
                # 提取真實 SST 和 Nino 指數
                SST = var_true1[:, :, self.sstlevel]
                nino_true1 = SST[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                
                # 模型預測
                out_var, residual_loss = self.mymodel(
                    input_var.float().to(self.device),
                    predictand=None,
                    train=False,
                )
                
                # 提取預測的 SST 和 Nino 指數
                SST_out = out_var[:, :, self.sstlevel]
                out_nino = SST_out[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                
                # 收集預測和真實值
                var_true.append(var_true1)
                nino_true.append(nino_true1)
                var_pred.append(out_var)
                nino_pred.append(out_nino)
            
            # 拼接所有批次的結果
            var_pred = torch.cat(var_pred, dim=0)
            nino_pred = torch.cat(nino_pred, dim=0)
            nino_true = torch.cat(nino_true, dim=0)
            var_true = torch.cat(var_true, dim=0)
            
            # 計算評估指標
            ninosc = self.calscore(nino_pred, nino_true.float().to(self.device))
            loss_var = self.loss_var(var_pred, var_true.float().to(self.device), residual_losses=None).item()
            loss_nino = self.loss_nino(
                nino_pred, nino_true.float().to(self.device)
            ).item()
            combine_loss = self.combine_loss(loss_var, loss_nino)
            
        return (
            var_pred,
            nino_pred,
            loss_var,
            loss_nino,
            combine_loss,
            ninosc,
        )

    def train_model(self, dataset_train, dataset_eval):
        """訓練模型"""
        # 根據激活函數配置修改模型保存路徑
        model_name = self.get_model_name()
        chk_path = self.mypara.model_savepath + model_name
        torch.manual_seed(self.mypara.seeds)
        
        # 創建數據加載器
        dataloader_train = DataLoader(
            dataset_train, batch_size=self.mypara.batch_size_train, shuffle=False
        )
        dataloader_eval = DataLoader(
            dataset_eval, batch_size=self.mypara.batch_size_eval, shuffle=False
        )
        
        count = 0
        best = -math.inf
        global_step = 0

        print(f"使用 Teacher Forcing 調度策略: {self.tf_scheduler.strategy}")
        print(f"使用混合激活函數配置: {self.activation_config}")
        print(f"模型將保存到: {chk_path}")

        mlflow.set_tracking_uri("http://localhost:5001")
        # 根據激活函數配置設置實驗名稱
        if isinstance(self.activation_config, str):
            experiment_name = f"GeoVMRNN_Mixed_{self.activation_config}"
        else:
            experiment_name = "GeoVMRNN_Mixed_Custom"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            mlflow.set_tag("model", "GeoVMRNN_MixedActivation")
            mlflow.set_tag("activation_config", str(self.activation_config))
            mlflow.log_params({
                "activation_config": str(self.activation_config),
                "lr_factor": self.opt.factor,
                "warmup": self.opt.warmup,
                "d_model": self.mypara.d_size,
                "batch_size": self.mypara.batch_size_train,
                "epochs": self.mypara.num_epochs,
                "tf_strategy": self.tf_scheduler.strategy
            })
        
            for i_epoch in range(self.mypara.num_epochs):
                print("==========" * 8)
                print(f"\n-->epoch: {i_epoch}")
                
                # 訓練階段
                self.mymodel.train()
                for j, (input_var, var_true) in enumerate(dataloader_train):
                    # 提取真實 SST 和 Nino 指數
                    SST = var_true[:, :, self.sstlevel]
                    nino_true = SST[
                        :,
                        :,
                        self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                        self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                    ].mean(dim=[2, 3])
                    
                    # 獲取當前的 Teacher Forcing 比例
                    current_tf_ratio = self.tf_scheduler.get_ratio()
                    
                    # 前向傳播
                    var_pred, residual_loss = self.mymodel(
                        input_var.float().to(self.device),
                        var_true.float().to(self.device),
                        train=True,
                        sv_ratio=current_tf_ratio,
                    )
                    
                    # 提取預測的 SST 和 Nino 指數
                    SST_pred = var_pred[:, :, self.sstlevel]
                    nino_pred = SST_pred[
                        :,
                        :,
                        self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                        self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                    ].mean(dim=[2, 3])
                    
                    # 計算損失
                    self.opt.optimizer.zero_grad()
                    loss_var = self.loss_var(var_pred, var_true.float().to(self.device), residual_loss)
                    loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
                    score = self.calscore(nino_pred, nino_true.float().to(self.device))
                    combine_loss = self.combine_loss(loss_var, loss_nino)
                    
                    # 反向傳播
                    combine_loss.backward()
                    self.opt.step()

                    # 更新 Teacher Forcing 調度器
                    if self.tf_scheduler.strategy == 'adaptive':
                        # 自适应调度需要提供当前性能分数
                        self.tf_scheduler.step(score)
                    else:
                        # 其他调度策略不需要性能分数
                        self.tf_scheduler.step()
                        
                    mlflow.log_metric("Train/Loss_Var", loss_var.item(), step=global_step)
                    mlflow.log_metric("Train/Loss_Nino", loss_nino.item(), step=global_step)
                    mlflow.log_metric("Train/Combine_Loss", combine_loss.item(), step=global_step)
                    mlflow.log_metric("Train/Score", score, step=global_step)
                    mlflow.log_metric("Train/tf_ratio", current_tf_ratio, step=global_step)
                    mlflow.log_metric("Train/Loss_Residual", residual_loss.item(), step=global_step)
                    global_step += 1

                    # 打印訓練進度
                    if j % 100 == 0:
                        print(
                            f"\n-->batch:{j} loss_var:{loss_var:.2f}, loss_nino:{loss_nino:.2f}, score:{score:.3f}, mixed_activation"
                        )

                    # 密集驗證
                    if (i_epoch + 1 >= 4) and (j + 1) % 200 == 0:
                        (
                            _,
                            _,
                            lossvar_eval,
                            lossnino_eval,
                            comloss_eval,
                            sceval,
                        ) = self.model_pred(dataloader=dataloader_eval)
                        
                        print(
                            f"-->Evaluation... \nloss_var:{lossvar_eval:.3f} \nloss_nino:{lossnino_eval:.3f} \nloss_com:{comloss_eval:.3f} \nscore:{sceval:.3f}"
                        )
                        mlflow.log_metric("Eval/Loss_Var", lossvar_eval, step=global_step)
                        mlflow.log_metric("Eval/Loss_Nino", lossnino_eval, step=global_step)
                        mlflow.log_metric("Eval/Combine_Loss", comloss_eval, step=global_step)
                        mlflow.log_metric("Eval/Score", sceval, step=global_step)

                        if sceval > best:
                            torch.save(self.mymodel.state_dict(), chk_path)
                            best = sceval
                            count = 0
                            print(f"\nsaving model with mixed activation...")
                
                # 每個 epoch 結束後的評估
                (
                    _,
                    _,
                    lossvar_eval,
                    lossnino_eval,
                    comloss_eval,
                    sceval,
                ) = self.model_pred(dataloader=dataloader_eval)
                
                print(
                    f"\n-->epoch{i_epoch} end... \nloss_var:{lossvar_eval:.3f} \nloss_nino:{lossnino_eval:.3f} \nloss_com:{comloss_eval:.3f} \nscore: {sceval:.3f}"
                )
                
                mlflow.log_metric("Epoch/Loss_Var", lossvar_eval, step=i_epoch)
                mlflow.log_metric("Epoch/Loss_Nino", lossnino_eval, step=i_epoch)
                mlflow.log_metric("Epoch/Combine_Loss", comloss_eval, step=i_epoch)
                mlflow.log_metric("Epoch/Score", sceval, step=i_epoch)

                # 檢查是否需要保存模型
                if sceval <= best:
                    count += 1
                    print(f"\nsc is not increase for {count} epoch")
                else:
                    count = 0
                    print(
                        f"\nsc is increase from {best:.3f} to {sceval:.3f} with mixed activation \nsaving model...\n"
                    )
                    torch.save(self.mymodel.state_dict(), chk_path)
                    best = sceval
                
                # 早停檢查
                if count == self.mypara.patience:
                    print(
                        f"\n-----!!!early stopping reached, max(sceval)= {best:.3f} with mixed activation!!!-----"
                    )
                    break
        
        del self.mymodel


def train_with_mixed_activation(activation_config=None, config_name="custom"):
    """使用指定混合激活函數配置訓練模型"""
    print(f"\n{'='*60}")
    print(f"開始訓練使用 {config_name} 混合激活函數配置的模型")
    print(f"激活函數配置: {activation_config}")
    print(f"{'='*60}")
    
    print(mypara.__dict__)
    print(f"\nloading pre-train dataset for {config_name} mixed activation model...")
    traindataset = make_dataset2(mypara)
    print(traindataset.selectregion())
    
    print(f"\nloading evaluation dataset for {config_name} mixed activation model...")
    evaldataset = make_testdataset(mypara, ngroup=100)
    print(evaldataset.selectregion())
    
    # 創建訓練器並開始訓練
    trainer = GeoVMRNNMixedActivationTrainer(mypara, activation_config=activation_config)
    trainer.train_model(
        dataset_train=traindataset,
        dataset_eval=evaldataset,
    )
    
    print(f"\n{config_name} 混合激活函數配置模型訓練完成！")


if __name__ == "__main__":
    # 獲取示例配置
    configs_to_train = get_ablation_configs()
    
    for config_name in configs_to_train:
        try:
            train_with_mixed_activation(config_name)
        except Exception as e:
            print(f"訓練 {config_name} 混合激活函數配置模型時出錯: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n所有混合激活函數配置模型訓練完成!")
