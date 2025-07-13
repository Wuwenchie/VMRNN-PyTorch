from GeoVMRNN import GeoVMRNN
from myconfig import mypara
import torch
from torch.utils.data import DataLoader
import numpy as np
import math
from LoadData import make_dataset2, make_testdataset


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


class GeoVMRNNTrainer:
    """GeoVMRNN 模型訓練器"""
    def __init__(self, mypara):
        assert mypara.input_channal == mypara.output_channal
        self.mypara = mypara
        self.device = mypara.device
        
        # 創建 GeoVMRNN 模型
        self.mymodel = GeoVMRNN_Supervised(mypara).to(mypara.device)
        
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

    def loss_var(self, y_pred, y_true):
        """計算變量損失"""
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
        rmse = rmse.sqrt().mean(dim=0)
        rmse = torch.sum(rmse, dim=[0, 1])
        return rmse

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
                out_var = self.mymodel(
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
            loss_var = self.loss_var(var_pred, var_true.float().to(self.device)).item()
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
        chk_path = self.mypara.model_savepath + "GeoVMRNN.pkl"
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
        sv_ratio = 1  # 用於調度視覺變分的比例
        
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
                
                # 調整視覺變分比例
                if sv_ratio > 0:
                    sv_ratio = max(sv_ratio - 2.5e-4, 0)
                
                # 前向傳播
                var_pred = self.mymodel(
                    input_var.float().to(self.device),
                    var_true.float().to(self.device),
                    train=True,
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
                loss_var = self.loss_var(var_pred, var_true.float().to(self.device))
                loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
                score = self.calscore(nino_pred, nino_true.float().to(self.device))
                combine_loss = self.combine_loss(loss_var, loss_nino)
                
                # 反向傳播
                combine_loss.backward()
                self.opt.step()
                
                # 打印訓練進度
                if j % 100 == 0:
                    print(
                        f"\n-->batch:{j} loss_var:{loss_var:.2f}, loss_nino:{loss_nino:.2f}, score:{score:.3f}"
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
                    
                    if sceval > best:
                        torch.save(self.mymodel.state_dict(), chk_path)
                        best = sceval
                        count = 0
                        print("\nsaving model...")
            
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
            
            # 檢查是否需要保存模型
            if sceval <= best:
                count += 1
                print(f"\nsc is not increase for {count} epoch")
            else:
                count = 0
                print(
                    f"\nsc is increase from {best:.3f} to {sceval:.3f}   \nsaving model...\n"
                )
                torch.save(self.mymodel.state_dict(), chk_path)
                best = sceval
            
            # 早停檢查
            if count == self.mypara.patience:
                print(
                    f"\n-----!!!early stopping reached, max(sceval)= {best:.3f}!!!-----"
                )
                break
        
        del self.mymodel


if __name__ == "__main__":
    print(mypara.__dict__)
    print("\nloading pre-train dataset...")
    traindataset = make_dataset2(mypara)
    print(traindataset.selectregion())
    
    print("\nloading evaluation dataset...")
    evaldataset = make_testdataset(mypara, ngroup=100)
    print(evaldataset.selectregion())
    
    # 創建訓練器並開始訓練
    trainer = GeoVMRNNTrainer(mypara)
    trainer.train_model(
        dataset_train=traindataset,
        dataset_eval=evaldataset,
    )
