from collections import defaultdict

import ray
import os
import re
from typing import List, Dict, Counter
import time

ray.init()

@ray.remote
def count_words_in_chunk(chunk: str) -> Dict[str, int]:
    words = re.findall(r'\b[a-zA-Z0-9]+\b', chunk.lower())
    return dict(Counter(words))

@ray.remote
def merge_results(*partial_results: List[Dict[str, int]]) -> Dict[str, int]:
    total_count = defaultdict(int)
    for result in partial_results:
        for word, count in result.items():
            total_count[word] += count
    return dict(total_count)

# 数据前置处理
def read_and_split_file(file_path: str, num_chunks: int = None) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()  # 按行读取

    if num_chunks is None:
        num_chunks = int(ray.available_resources().get("CPU", 4))
        num_chunks = max(1, num_chunks)

    # 每块包含若干完整行，避免切单词
    chunk_size = (len(lines) + num_chunks - 1) // num_chunks
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = ''.join(lines[i:i + chunk_size])
        chunks.append(chunk)

    print(f"文件 {file_path} 被分割成 {len(chunks)} 个块，用于并行处理")
    return chunks


def word_count(input_paths: List[str], output_path: str):
    start_time = time.time()

    all_futures = []

    for file_path in input_paths:
        chunks = read_and_split_file(file_path)
        # map阶段
        chunk_futures = [count_words_in_chunk.remote(chunk) for chunk in chunks]
        all_futures.extend(chunk_futures)

    print(f"time={time.time()} 已提交 {len(all_futures)} 个并行处理任务...")

    partial_results = ray.get(all_futures)
    print(f"time={time.time()} 所有 {len(partial_results)} 个任务已完成，开始合并结果....")

    # reduce阶段
    final_result_future = merge_results.remote(*partial_results)
    final_word_count = ray.get(final_result_future)
    sorted_word_count = sorted(final_word_count.items(), key=lambda x: x[1], reverse=True)

    with open(output_path, 'w', encoding="utf-8") as f:
        for word, count in sorted_word_count:
            f.write(f"{word}\t{count}\n")

    end_time = time.time()
    print(f"Wordcount完成！，结果保存至{output_path}")
    print(f"处理 {len(input_paths)} 个文件，共统计到 {len(final_word_count)} 个不同单词")
    print(f"总耗时： {end_time - start_time:.2f}秒")
    return sorted_word_count

if __name__ == "__main__":
    file1 = 'text1.txt'
    file2 = 'text2.txt'

    input_files = [file1, file2]
    output_file = 'word_count_output.txt'
    result = word_count(input_files, output_file)

    print("\n前10个高频词：")
    for word, count in result[:10]:
        print(f"{word}\t{count}")

    ray.shutdown()
