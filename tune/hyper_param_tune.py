import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 初始化 Ray 集群 (在单机上运行)
ray.init()


# 2. 定义你的机器学习模型训练和评估函数
def train_mnist(config):
    # 这个函数会在 Ray 的每个工作节点上独立运行
    # config 包含了本次运行的超参数
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, config["hidden_size"]),
        nn.ReLU(),
        nn.Linear(config["hidden_size"], 10)
    )

    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # 加载数据 (简化)
    transform = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)

    for epoch in range(10):  # 训练10个epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 向 Ray Tune 报告中间结果 (用于早停等)
            if batch_idx % 100 == 0:
                accuracy = tune.report(mean_accuracy=compute_accuracy(model))  # 假设有这个函数
                # Tune 会记录这个结果


# 3. 定义超参数搜索空间
config = {
    "lr": tune.loguniform(1e-4, 1e-1),  # 学习率，对数均匀分布
    "hidden_size": tune.choice([32, 64, 128, 256]),  # 隐藏层大小，从列表中选择
}

# 4. 定义调度器 (可选，用于早停低效的试验)
scheduler = ASHAScheduler(
    metric="mean_accuracy",
    mode="max",
    max_t=10,  # 最大训练epoch
    grace_period=1,  # 至少运行1个epoch才考虑早停
    reduction_factor=2  # 每次减少2倍的试验
)

# 5. 启动超参数搜索
analysis = tune.run(
    train_mnist,  # 要运行的训练函数
    config=config,  # 超参数搜索空间
    num_samples=100,  # 总共尝试100种不同的超参数组合
    scheduler=scheduler,  # 使用ASHA调度器进行早停
    resources_per_trial={"cpu": 1, "gpu": 0.5},  # 每个试验分配的资源
    # 如果有多个GPU，Ray会自动分配
)

# 6. 获取最佳结果
print("Best config: ", analysis.best_config)
print("Best accuracy: ", analysis.best_result["mean_accuracy"])

# 关闭 Ray
ray.shutdown()