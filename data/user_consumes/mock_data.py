import pandas as pd
import numpy as np
import os

# 生成 5 个示例文件用于测试
os.makedirs("data", exist_ok=True)

for i in range(5):
    df = pd.DataFrame({
        'user_id': np.random.choice([f'U{1000+j}' for j in range(500)], 10000),
        'product': np.random.choice(['laptop', 'mouse', 'keyboard', 'phone', 'tablet'], 10000),
        'price': np.random.uniform(10, 1500, 10000).round(2),
        'quantity': np.random.randint(1, 5, 10000),
        'timestamp': pd.date_range('2025-08-01', periods=10000, freq='T')
    })
    df.to_csv(f"data/sales_{i}.csv", index=False)