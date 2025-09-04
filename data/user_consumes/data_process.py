import ray

# 启动 Ray（本地模式）
ray.init()

# 1. 加载所有 CSV 文件 → 创建一个分布式 Dataset
ds = ray.data.read_csv(["data/sales_0.csv", "data/sales_1.csv", "data/sales_2.csv", "data/sales_3.csv", "data/sales_4.csv"])

print("数据集结构:")
# ds.show(3)  # 显示前 3 行

print(ds.count())

# 2. 添加新字段：每笔订单的总价
ds_with_total = ds.map(lambda row: {
    **row,
    "total": row['price'] * row['quantity']
})

# 3. 按 user_id 聚合，计算每个用户的总消费
aggregated_ds = ds_with_total.groupby("user_id").sum("total")

# 4. 按总消费排序，取前 10 名
top_users = aggregated_ds.sort("sum(total)", descending=True).limit(10)

# 5. 输出结果
print("\n消费最高的 10 位用户:")
for row in top_users.iter_rows():
    print(f"{row['user_id']}: ¥{row['sum(total)']:.2f}")