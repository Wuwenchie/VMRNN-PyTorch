import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from utils import set_seed, make_dir, init_logger, plot_loss
from configs import get_args
from functions import train_fusion, test_fusion
from LoadData import make_dataset1, make_dataset2, make_testdataset
from GeoformerVMRNN import GeoformerVMRNN


def setup_fusion_model(args):
    """設置融合模型"""
    # 創建模型參數對象
    class ModelParams:
        def __init__(self, args):
            self.device = args.device
            self.d_size = args.d_size
            self.input_channal = args.input_channels
            self.needtauxy = args.needtauxy
            self.patch_size = args.patch_size
            self.emb_spatial_size = args.emb_spatial_size
            self.input_length = args.input_length
            self.output_length = args.output_length
            self.nheads = args.nheads
            self.dim_feedforward = args.dim_feedforward
            self.dropout = args.dropout
            self.num_encoder_layers = args.num_encoder_layers
            self.num_decoder_layers = args.num_decoder_layers
    
    mypara = ModelParams(args)
    
    # 創建融合模型
    model = GeoformerVMRNN(mypara).to(args.device)
    
    # 加載預訓練權重（如果有）
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    
    # 優化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 損失函數
    criterion = nn.MSELoss()
    
    # 數據集設置
    if args.dataset_type == 'make_dataset1':
        train_dataset = make_dataset1(mypara)
        valid_dataset = make_testdataset(mypara, ngroup=args.valid_samples)
    elif args.dataset_type == 'make_dataset2':
        train_dataset = make_dataset2(mypara)
        valid_dataset = make_testdataset(mypara, ngroup=args.valid_samples)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    # 數據加載器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size,
        num_workers=args.num_workers, 
        shuffle=True, 
        pin_memory=True, 
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.valid_batch_size,
        num_workers=args.num_workers, 
        shuffle=False, 
        pin_memory=True, 
        drop_last=True
    )
    
    return model, criterion, optimizer, train_loader, valid_loader


def train_fusion(args, logger, epoch, model, train_loader, criterion, optimizer):
    """融合模型訓練函數"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data_x, data_y, *_) in enumerate(train_loader):
        # 移動到設備
        data_x = data_x.float().to(args.device)  # [B, input_length, C, H, W]
        data_y = data_y.float().to(args.device)  # [B, output_length, C, H, W]
        
        optimizer.zero_grad()
        
        # 前向傳播
        output = model(
            predictor=data_x, 
            predictand=data_y, 
            train=True, 
            sv_ratio=args.sv_ratio
        )
        
        # 計算損失
        loss = criterion(output, data_y)
        
        # 反向傳播
        loss.backward()
        
        # 梯度裁剪
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % args.log_interval == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / num_batches
    logger.info(f'Epoch {epoch} Training Loss: {avg_loss:.6f}')
    return avg_loss


def test_fusion(args, logger, epoch, model, valid_loader, criterion, cache_dir):
    """融合模型測試函數"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (data_x, data_y, *_) in enumerate(valid_loader):
            # 移動到設備
            data_x = data_x.float().to(args.device)
            data_y = data_y.float().to(args.device)
            
            # 前向傳播
            output = model(
                predictor=data_x, 
                predictand=None, 
                train=False
            )
            
            # 計算損失
            loss = criterion(output, data_y)
            mse = torch.mean((output - data_y) ** 2)
            
            total_loss += loss.item()
            total_mse += mse.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    
    # 計算SSIM（簡化版本）
    ssim = calculate_ssim(output, data_y)
    
    logger.info(f'Epoch {epoch} Validation Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, SSIM: {ssim:.6f}')
    return avg_loss, avg_mse, ssim


def calculate_ssim(pred, target):
    """計算SSIM（簡化版本）"""
    # 這裡提供一個簡化的SSIM計算
    # 實際使用時可以用專門的SSIM庫
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    
    pred_var = torch.var(pred)
    target_var = torch.var(target)
    
    covariance = torch.mean((pred - pred_mean) * (target - target_mean))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = (2 * pred_mean * target_mean + c1) * (2 * covariance + c2) / \
           ((pred_mean ** 2 + target_mean ** 2 + c1) * (pred_var + target_var + c2))
    
    return ssim.item()


def main():
    """主訓練函數"""
    # 獲取參數
    args = get_args()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 創建目錄
    cache_dir, model_dir, log_dir = make_dir(args)
    
    # 初始化日誌
    logger = init_logger(log_dir)
    
    # 設置模型
    model, criterion, optimizer, train_loader, valid_loader = setup_fusion_model(args)
    
    # 訓練記錄
    train_losses, valid_losses = [], []
    best_metric = (0, float('inf'), float('inf'))  # (epoch, mse, ssim)
    
    logger.info(f"Starting training with fusion model...")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 訓練循環
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 訓練
        train_loss = train_fusion(args, logger, epoch, model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        
        # 繪製訓練損失
        plot_loss(train_losses, 'train', epoch, args.res_dir, 1)
        
        # 驗證
        if (epoch + 1) % args.epoch_valid == 0:
            valid_loss, mse, ssim = test_fusion(args, logger, epoch, model, valid_loader, criterion, cache_dir)
            valid_losses.append(valid_loss)
            
            # 繪製驗證損失
            plot_loss(valid_losses, 'valid', epoch, args.res_dir, args.epoch_valid)
            
            # 保存最佳模型
            if mse < best_metric[1]:
                torch.save(model.state_dict(), f'{model_dir}/best_fusion_model.pth')
                best_metric = (epoch, mse, ssim)
                logger.info(f'New best model saved at epoch {epoch}')
            
            logger.info(f'[Current Best] EP:{best_metric[0]:04d} MSE:{best_metric[1]:.4f} SSIM:{best_metric[2]:.4f}')
        
        # 定期保存檢查點
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'best_metric': best_metric,
            }, f'{model_dir}/checkpoint_epoch_{epoch}.pth')
        
        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s')
    
    # 保存最終模型
    torch.save(model.state_dict(), f'{model_dir}/final_fusion_model.pth')
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
