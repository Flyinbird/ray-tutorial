import ray

# 启动 Ray（本地模式）
ray.init()

# 1. 加载所有 CSV 文件 → 创建一个分布式 Dataset
ds = ray.data.read_csv(["data/sales_0.csv", "data/sales_1.csv", "data/sales_2.csv", "data/sales_3.csv", "data/sales_4.csv"])

print("数据集结构:")
ds.show(3)  # 显示前 3 行

print(ds.count())

# 2. 添加新字段：每笔订单的总价
top_users = (ds.map(lambda row: {
    **row,
    "total": row['price'] * row['quantity']
    })
    .groupby("user_id")
    .sum("total")
    .sort("sum(total)", descending=True)
    .limit(10))


# 3. 输出结果
print("\n消费最高的 10 位用户:")
for row in top_users.iter_rows():
    print(f"{row['user_id']}: ¥{row['sum(total)']:.2f}")