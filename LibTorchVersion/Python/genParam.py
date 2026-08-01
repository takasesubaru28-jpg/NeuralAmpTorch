from scipy.stats import qmc
import pandas as pd
import numpy as np

def generate_lhs_amp_params(num_samples, param_names):
    dim = len(param_names)
    # 1. LHSサンプラーの初期化（seedを設定して再現性を確保）
    sampler = qmc.LatinHypercube(d=dim, seed=42)
    sample = sampler.random(n=num_samples)
    
    # 2. 0.0 ~ 10.0 の範囲にスケーリング
    l_bounds = [0.0] * dim
    u_bounds = [10.0] * dim
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
    
    # 3. 有効数字2桁（小数点第1位）で丸める
    # アンプのノブ設定として現実的な 0.0, 0.1 ... 10.0 に変換
    rounded_sample = np.round(scaled_sample, 1)
    
    df = pd.DataFrame(rounded_sample, columns=param_names)
    return df

# 設定例：30パターンの録音設定を作る
params = ['Bass', 'Middle', 'Treble', 'Gain']
df_configs = generate_lhs_amp_params(1000, params)

# 重複を排除（丸め処理で稀に被ることがあるため）
df_configs = df_configs.drop_duplicates()

# CSV保存
df_configs.to_csv("lhs_amp_settings.csv",index=None)

print(f"生成された設定数: {len(df_configs)}")
print(df_configs.head(10))