import argparse
import torch

def get_fusion_args():
    """獲取融合模型的配置參數"""
    parser = argparse.ArgumentParser(description='Geoformer-VMRNN Fusion Model Training')
    
    # 基本設置
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    
    # 數據設置
    parser.add_argument('--dataset_type', type=str, default='make_dataset1',
                        choices=['make_dataset1', 'make_dataset2'],
                        help='Dataset type to use')
    parser.add_argument('--input_channels', type=int, default=23,
                        help='Number of input channels (temperature levels)')
    parser.add_argument('--needtauxy', action='store_true',
                        help='Whether to include tau_x and tau_y fields')
    parser.add_argument('--input_length', type=int, default=12,
                        help='Input sequence length')
    parser.add_argument('--output_length', type=int, default=6,
                        help='Output sequence length')
    
    # 模型結構參數
    parser.add_argument('--d_size', type=int, default=512,
                        help='Transformer model dimension')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[4, 4],
                        help='Patch size for tokenization')
    parser.add_argument('--emb_spatial_size', type=int, nargs=2, default=[32, 64],
                        help='Spatial embedding size [lat, lon]')
    parser.add_argument('--nheads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                        help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                        help='Number of decoder layers')
    
    # 訓練參數
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--valid_batch_size', type=int, default=16,
                        help='Validation batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--sv_ratio', type=float, default=0.1,
                        help='Supervision ratio for scheduled sampling')
    
    # 驗證和保存
    parser.add_argument('--epoch_valid', type=int, default=5,
                        help='Validation frequency (epochs)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Model saving frequency (epochs)')
    parser.add_argument('--valid_samples', type=int, default=1000,
                        help='Number of validation samples')
    
    # 梯度裁剪
    parser.add_argument('--clip_grad', action='store_true',
                        help='Whether to clip gradients')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm')
    
    # 路徑設置
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--res_dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging frequency (batches)')
    
    # 數據路徑（需要根據您的實際路徑修改）
    parser.add_argument('--adr_pretr', type=str, 
                        default='/path/to/your/training/data.nc',
                        help='Path to training data')
    parser.add_argument('--adr_eval', type=str,
                        default='/path/to/your/evaluation/data.nc',
                        help='Path to evaluation data')
    
    # 數據範圍設置
    parser.add_argument('--lev_range', type=int, nargs=2, default=[0, 23],
                        help='Level range for temperature data')
    parser.add_argument('--lat_range', type=int, nargs=2, default=[0, 128],
                        help='Latitude range')
    parser.add_argument('--lon_range', type=int, nargs=2, default=[0, 256],
                        help='Longitude range')
    
    # 在線數據集參數
    parser.add_argument('--all_group', type=int, default=10000,
                        help='Total number of samples for online dataset')
    parser.add_argument('--look_back', type=int, default=12,
                        help='Look back window for dataset1')
    parser.add_argument('--pre_len', type=int, default=6,
                        help='Prediction length for dataset1')
    parser.add_argument('--interval', type=
