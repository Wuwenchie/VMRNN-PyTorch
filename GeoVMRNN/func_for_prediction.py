# from GeoVMRNN_snake import GeoVMRNN_MixedActivation
from GeoVMRNN_ablation import GeoVMRNN_Supervised
# from GeoVMRNN_sv import GeoVMRNN_Supervised
import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr
from scipy.signal.windows import hann
import matplotlib as mpl


class make_dataset_test(Dataset):
    def __init__(
        self,
        address,
        needtauxy,
        lev_range=(0, 1),
        lon_range=(0, 1),
        lat_range=(0, 1),
    ):
        data_in = xr.open_dataset(address)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = lev_range
        self.lon_range = lon_range
        self.lat_range = lat_range

        temp = data_in["temperatureNor"][
            :,
            :,
            lev_range[0] : lev_range[1],
            lat_range[0] : lat_range[1],
            lon_range[0] : lon_range[1],
        ].values
        temp = np.nan_to_num(temp)
        temp[abs(temp) > 999] = 0
        if needtauxy:
            taux = data_in["tauxNor"][
                :,
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :,
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0
            # --------------
            self.dataX = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )
            del temp, taux, tauy
        else:
            self.dataX = temp
            del temp

    def getdatashape(self):
        return {
            "dataX.shape": self.dataX.shape,
        }

    def selectregion(self):
        return {
            "lon: {}E to {}E".format(
                self.lon[self.lon_range[0]],
                self.lon[self.lon_range[1] - 1],
            ),
            "lat: {}S to {}N".format(
                self.lat[self.lat_range[0]],
                self.lat[self.lat_range[1] - 1],
            ),
            "lev: {}m to {}m".format(
                self.lev[self.lev_range[0]], self.lev[self.lev_range[1] - 1]
            ),
        }

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx]


def load_model_with_flexible_activation(model, checkpoint_path, strict=False):
    """
    靈活載入模型，處理激活函數參數不匹配的問題
    
    Args:
        model: 已初始化的模型實例
        checkpoint_path: 檢查點文件路徑
        strict: 是否嚴格載入（False允許忽略缺失的鍵）
    """
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    
    if not strict:
        # 獲取當前模型的狀態字典
        model_state = model.state_dict()
        
        # 過濾檢查點中與當前模型匹配的參數
        filtered_checkpoint = {}
        missing_keys = []
        unexpected_keys = []
        
        for key, value in checkpoint.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered_checkpoint[key] = value
                else:
                    print(f"形狀不匹配，跳過參數: {key}")
                    print(f"  模型形狀: {model_state[key].shape}")
                    print(f"  檢查點形狀: {value.shape}")
            else:
                unexpected_keys.append(key)
        
        # 檢查缺失的鍵
        for key in model_state:
            if key not in filtered_checkpoint:
                missing_keys.append(key)
        
        # 載入過濾後的參數
        model.load_state_dict(filtered_checkpoint, strict=False)
        
        if missing_keys:
            print(f"警告：以下參數將使用隨機初始化: {missing_keys}")
        if unexpected_keys:
            print(f"警告：以下參數在檢查點中但不在模型中: {unexpected_keys}")
            
        return model, missing_keys, unexpected_keys
    else:
        model.load_state_dict(checkpoint, strict=True)
        return model, [], []


def func_pre(mypara, adr_model, adr_datain, adr_oridata, needtauxy, 
             activation_config=None, strict_loading=False):
    """
    修改後的預測函數，支持靈活的模型載入
    
    Args:
        mypara: 模型參數
        adr_model: 模型檢查點路徑
        adr_datain: 輸入數據地址
        adr_oridata: 原始數據地址
        needtauxy: 是否需要tau數據
        activation_config: 激活函數配置（如果為None，嘗試匹配檢查點）
        strict_loading: 是否嚴格載入模型參數
    """
    lead_max = mypara.output_length
    
    # 數據載入部分保持不變
    data_ori = xr.open_dataset(adr_oridata)
    temp_ori_region = data_ori["temperatureNor"][
        :,
        mypara.lev_range[0] : mypara.lev_range[1],
        mypara.lat_range[0] : mypara.lat_range[1],
        mypara.lon_range[0] : mypara.lon_range[1],
    ].values
    nino34 = data_ori["nino34"].values
    stdtemp = data_ori["stdtemp"][mypara.lev_range[0] : mypara.lev_range[1]].values
    stdtemp = np.nanmean(stdtemp, axis=(1, 2))
    
    if needtauxy:
        taux_ori_region = data_ori["tauxNor"][
            :,
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        tauy_ori_region = data_ori["tauyNor"][
            :,
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        stdtaux = data_ori["stdtaux"].values
        stdtaux = np.nanmean(stdtaux, axis=(0, 1))
        stdtauy = data_ori["stdtauy"].values
        stdtauy = np.nanmean(stdtauy, axis=(0, 1))

        var_ori_region = np.concatenate(
            (taux_ori_region[:, None], tauy_ori_region[:, None], temp_ori_region),
            axis=1,
        )
        del taux_ori_region, tauy_ori_region, temp_ori_region
        stds = np.concatenate((stdtaux[None], stdtauy[None], stdtemp), axis=0)
        del stdtemp, stdtauy, stdtaux
    else:
        var_ori_region = temp_ori_region
        del temp_ori_region
        stds = stdtemp
        del stdtemp

    # 數據集載入
    dataCS = make_dataset_test(
        address=adr_datain,
        needtauxy=needtauxy,
        lev_range=mypara.lev_range,
        lon_range=mypara.lon_range,
        lat_range=mypara.lat_range,
    )
    test_group = len(dataCS)
    print(dataCS.getdatashape())
    print(dataCS.selectregion())
    dataloader_test = DataLoader(
        dataCS, batch_size=mypara.batch_size_eval, shuffle=False
    )

    activation_config = {
        'cnn': 'relu',
        'fusion': 'relu',
        'prediction': 'relu'
    }
    # 根據是否提供激活函數配置來處理模型初始化
    if activation_config is None:
        # 方案1：使用默認配置（假設檢查點是用ReLU訓練的）
        print("使用默認激活函數配置 (ReLU)")
        activation_config = 'relu'
    else:
        print(f"使用自定義激活函數配置: {activation_config}")

    # 創建模型
    mymodel = GeoVMRNN_Supervised(mypara, activation_config=activation_config).to(mypara.device)
    # mymodel = GeoVMRNN_Supervised(mypara)
    # mymodel = GeoVMRNN_Supervised(mypara).to(mypara.device)
    
    # 靈活載入模型
    try:
        if strict_loading:
            mymodel.load_state_dict(torch.load(adr_model))
            print("嚴格模式載入成功")
        else:
            mymodel, missing_keys, unexpected_keys = load_model_with_flexible_activation(
                mymodel, adr_model, strict=False
            )
            print("靈活模式載入完成")
    except Exception as e:
        print(f"模型載入失敗: {e}")
        raise e

    mymodel.eval()
    torch.set_grad_enabled(False)
    
    # 其餘部分保持不變
    if needtauxy:
        n_lev = mypara.lev_range[1] - mypara.lev_range[0] + 2
        sst_lev = 2
    else:
        n_lev = mypara.lev_range[1] - mypara.lev_range[0]
        sst_lev = 0
        
    var_pred = np.zeros(
        [
            test_group,
            lead_max,
            n_lev,
            mypara.lat_range[1] - mypara.lat_range[0],
            mypara.lon_range[1] - mypara.lon_range[0],
        ]
    )
    
    ii = 0
    iii = 0
    # with torch.no_grad():
    #     for input_var in dataloader_test:
    #         out_var = mymodel(
    #             input_var.float().to(mypara.device),
    #             predictand=None,
    #             train=False,
    #         )
    #         ii += out_var.shape[0]
    #         if torch.cuda.is_available():
    #             var_pred[iii:ii] = out_var.cpu().detach().numpy()
    #         else:
    #             var_pred[iii:ii] = out_var.detach().numpy()
    #         iii = ii
    with torch.no_grad():
        for input_var in dataloader_test:
            out_var = mymodel(
                input_var.float().to(mypara.device),
                predictand=None,
                train=False,
            )
            # 如果模型返回元組，取第一個元素（通常是預測結果）
            if isinstance(out_var, tuple):
                out_var = out_var[0]
            
            ii += out_var.shape[0]
            if torch.cuda.is_available():
                var_pred[iii:ii] = out_var.cpu().detach().numpy()
            else:
                var_pred[iii:ii] = out_var.detach().numpy()
            iii = ii
            
    del out_var, input_var
    del mymodel, dataCS, dataloader_test

    # 數據處理部分保持不變
    len_data = test_group - lead_max
    print("len_data:", len_data)
    
    start_idx = 12+lead_max-1
    cut_var_true = var_ori_region[start_idx : start_idx + len_data]
    cut_var_true = cut_var_true * stds[None, :, None, None]
    cut_nino_true = nino34[start_idx : start_idx + len_data]
    
    print('cut_nino_true:', cut_nino_true.shape[0])
    print('cut_var_true:', cut_var_true.shape[0])
    assert cut_nino_true.shape[0] == cut_var_true.shape[0] == len_data
    
    cut_var_pred = np.zeros(
        [lead_max, len_data, var_pred.shape[2], var_pred.shape[3], var_pred.shape[4]]
    )
    cut_nino_pred = np.zeros([lead_max, len_data])
    
    for i in range(lead_max):
        l = i + 1
        cut_var_pred[i] = (
            var_pred[lead_max - l : test_group - l, i] * stds[None, :, None, None]
        )
        cut_nino_pred[i] = np.nanmean(
            cut_var_pred[
                i,
                :,
                sst_lev,
                mypara.lat_nino_relative[0] : mypara.lat_nino_relative[1],
                mypara.lon_nino_relative[0] : mypara.lon_nino_relative[1],
            ],
            axis=(1, 2),
        )
    
    assert cut_var_pred.shape[1] == cut_var_true.shape[0]
    return (
        cut_var_pred,
        cut_var_true,
        cut_nino_pred,
        cut_nino_true,
    )



# 方案2：使用靈活載入，允許部分參數缺失
# def example_usage_2(mypara, adr_model, adr_datain, adr_oridata, needtauxy):
#     """使用Snake激活函數，但允許參數缺失（會隨機初始化缺失的激活函數參數）"""
#     activation_config = {
#         'cnn': 'relu',
#         'fusion': 'relu',
#         'prediction': 'relu'
#     }
#     return func_pre(
#         mypara, adr_model, adr_datain, adr_oridata, needtauxy,
#         activation_config=activation_config,
#         strict_loading=False  # 允許參數缺失
#     )
