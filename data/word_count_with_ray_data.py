import ray
import re
from typing import Dict, List

# 初始化 Ray
ray.init(ignore_reinit_error=True)


def wordcount_with_ray_data(input_paths: List[str], output_path: str):
    """
    使用 Ray Data 实现分布式 WordCount
    :param input_paths: 输入文件路径列表
    :param output_path: 输出文件路径
    """

    print(f"正在从 {input_paths} 加载数据...")

    # Step 1: 读取文本文件
    ds = ray.data.read_text(input_paths)

    ds = ds.materialize()
    # 查看数据集信息
    print(f"数据集已加载，共 {ds.count()} 行，有 {ds.num_blocks()} 个分片。")

    # Step 2: 分词 - 将每行文本拆分为单词
    def split_lines(batch: Dict[str, List]) -> Dict[str, List]:
        """将每行文本拆分为单词"""
        words_list = []
        for line in batch["text"]:
            # 使用正则表达式提取单词，转小写
            words = re.findall(r'\b[a-zA-Z]+\b', line.lower())
            words_list.append(words)
        return {"words": words_list}

    # 执行分词操作
    word_ds = ds.map_batches(
        split_lines,
        batch_size=1000,
        batch_format="pandas"
    )

    # Step 3: 扁平化 - 将单词列表展开
    def flatten_words(batch: Dict[str, List]) -> Dict[str, List]:
        """将嵌套的单词列表展开为扁平列表"""
        all_words = []
        for words in batch["words"]:
            all_words.extend(words)
        return {"word": all_words}

    flat_ds = word_ds.map_batches(
        flatten_words,
        batch_size=1000,
        batch_format="pandas"
    )

    # Step 4: 统计词频 - 使用 Ray Data 的内置聚合功能
    print("开始统计词频...")

    # 使用 groupby 进行统计（推荐）
    word_count_ds = flat_ds.groupby("word").count()

    # 转换为字典格式
    word_count_dict = {}
    for batch in word_count_ds.iter_batches():
        for word, count in zip(batch["word"], batch["count()"]):
            word_count_dict[word] = count

    print("词频统计完成！")

    # Step 5: 排序并写入文件
    sorted_word_count = sorted(
        word_count_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # 写入结果文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for word, count in sorted_word_count:
            f.write(f"{word}\t{count}\n")

    print(f"结果已保存至: {output_path}")
    print(f"共统计到 {len(word_count_dict)} 个不同单词")
    print(f"总单词数: {sum(word_count_dict.values())}")

    return word_count_dict



if __name__ == "__main__":

    # 输入输出路径
    input_files = ["text1.txt", "text2.txt"]
    output_file = "word_count_output_with_ray_data.txt"
    try:
        # 方法1: 完整实现
        print("=" * 50)
        print("方法1: 完整版 WordCount")
        print("=" * 50)
        result1 = wordcount_with_ray_data(input_files, output_file)

        # 显示前10个高频词
        print("\n前10个高频词 (完整版):")
        sorted_result = sorted(result1.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_result[:10]:
            print(f"  {word}: {count}")

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 清理
        ray.shutdown()
        print("Ray 已关闭")