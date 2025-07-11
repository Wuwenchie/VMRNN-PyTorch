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
    
    # VMRNN 特定參數
    parser.add_argument('--vmrnn_hidden_size', type=int, default=128,
                        help='VMRNN hidden state size')
    parser.add_argument('--vmrnn_num_layers', type=int, default=2,
                        help='Number of VMRNN layers')
    parser.add_argument('--vmrnn_dropout', type=float, default=0.1,
                        help='VMRNN dropout rate')
    
    # 融合模型參數
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'add', 'attention'],
                        help='Fusion method for combining Geoformer and VMRNN')
    parser.add_argument('--fusion_hidden_size', type=int, default=256,
                        help='Hidden size for fusion layer')
    parser.add_argument('--geoformer_weight', type=float, default=0.5,
                        help='Weight for Geoformer in fusion (0-1)')
    parser.add_argument('--vmrnn_weight', type=float, default=0.5,
                        help='Weight for VMRNN in fusion (0-1)')
    
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
    
    # 學習率調度
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['step', 'cosine', 'plateau'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    
    # 驗證和保存
    parser.add_argument('--epoch_valid', type=int, default=5,
                        help='Validation frequency (epochs)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Model saving frequency (epochs)')
    parser.add_argument('--valid_samples', type=int, default=1000,
                        help='Number of validation samples')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience (epochs)')
    
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
    parser.add_argument('--tensorboard_log', type=str, default='./logs',
                        help='Tensorboard log directory')
    
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
    parser.add_argument('--interval', type=int, default=1,
                        help='Sampling interval for dataset')
    
    # 數據預處理參數
    parser.add_argument('--normalize', action='store_true',
                        help='Whether to normalize input data')
    parser.add_argument('--data_augmentation', action='store_true',
                        help='Whether to apply data augmentation')
    parser.add_argument('--noise_level', type=float, default=0.01,
                        help='Noise level for data augmentation')
    
    # 損失函數參數
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'mae', 'huber', 'combined'],
                        help='Loss function type')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.0],
                        help='Weights for different loss components')
    
    # 評估指標
    parser.add_argument('--eval_metrics', type=str, nargs='+',
                        default=['mse', 'mae', 'rmse', 'mape'],
                        help='Evaluation metrics to compute')
    
    # 實驗設置
    parser.add_argument('--experiment_name', type=str, default='fusion_experiment',
                        help='Experiment name for logging')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name for this specific experiment')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    # 多GPU訓練
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    # 混合精度訓練
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--grad_scale', type=float, default=1.0,
                        help='Gradient scaling factor for mixed precision')
    
    return parser

def validate_args(args):
    """驗證參數的有效性"""
    # 確保權重加起來等於1
    if abs(args.geoformer_weight + args.vmrnn_weight - 1.0) > 1e-6:
        print("Warning: Geoformer and VMRNN weights should sum to 1.0")
        # 自動調整
        total_weight = args.geoformer_weight + args.vmrnn_weight
        args.geoformer_weight /= total_weight
        args.vmrnn_weight /= total_weight
    
    # 確保批次大小是合理的
    if args.train_batch_size <= 0 or args.valid_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    
    # 確保路徑存在或創建
    import os
    os.makedirs(args.res_dir, exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)
    
    return args

if __name__ == '__main__':
    parser = get_fusion_args()
    args = parser.parse_args()
    args = validate_args(args)
    
    print("Fusion Model Configuration:")
    print("-" * 50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
