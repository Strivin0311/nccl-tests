import os
import pandas as pd


algos_of_coll = {
  "AllGather":     ["RING","COLLNET_DIRECT","NVLS","PAT"],
  "AllReduce":     ["TREE","RING","COLLNET_DIRECT","COLLNET_CHAIN","NVLS","NVLS_TREE"],
  "Broadcast":     ["RING"],
  "Reduce":        ["RING"],
  "ReduceScatter": ["RING","COLLNET_DIRECT","NVLS","PAT"],
  "SendRecv":      [None]
}

coll_camel_to_lower = {
  "AllGather":     "all_gather",
  "AllReduce":     "all_reduce",
  "Broadcast":     "broadcast",
  "Reduce":        "reduce",
  "ReduceScatter": "reduce_scatter",
  "SendRecv":      "sendrecv"
}

def process_nccl_output(file_path, peak_bw):
    """
    处理 NCCL 测试输出文件，将其转换为 Pandas DataFrame，并添加 bwu 列。
    
    :param file_path: NCCL 测试输出文件的路径
    :param peak_bw: 峰值带宽 (GB/s)
    :return: 更新后的 Pandas DataFrame
    """
    # 读取文件到字符串
    with open(file_path, 'r') as file:
        output = file.read()

    # 将输出转换为 DataFrame
    lines = output.splitlines()
    data = []
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        columns = line.split()
        if len(columns) < 12:
            continue
        data.append(columns)

    # 定义列名
    column_names = ['size', 'count', 'type', 'redop', 'root', 'time', 'algbw', 'busbw', 'wrong', 'time_inplace', 'algbw_inplace', 'busbw_inplace', 'wrong_inplace']

    # 创建 DataFrame
    df = pd.DataFrame(data, columns=column_names)

    # 计算 bwu
    df['algbw'] = df['algbw'].astype(float)
    df['busbw'] = df['busbw'].astype(float)
    
    # bwu w.r.t. busbw, however, for comm using SHArP/Collnet like AllReduce,
    # the busbw is much harder to to make sense of and we usually prefer the algbw
    # see issue: https://github.com/NVIDIA/nccl/issues/320
    df['bwu'] = (df['busbw'] / peak_bw) * 100 
    
    # 删除不需要的列
    df.drop(columns=['count', 'type', 'redop', 'root', 'time_inplace', 'wrong', 'wrong_inplace', 'algbw_inplace', 'busbw_inplace'], inplace=True)
    
    # 添加单位
    df['size'] = df['size'].astype(int)
    df['size'] = df['size'].apply(lambda x: f"{x / 1024**3:.1f}GB" if x >= 1024**3 else f"{x / 1024**2:.1f}MB")
    df.rename(columns={'size': 'comm_volume'}, inplace=True)
    df.rename(columns={'time': 'duration (μs)'}, inplace=True)
    df.rename(columns={'algbw': 'algbw (GB/s)'}, inplace=True)
    df.rename(columns={'busbw': 'busbw (GB/s)'}, inplace=True)
    
    df['bwu'] = df['bwu'].apply(lambda x: f"{x:.2f}")
    df.rename(columns={'bwu': 'bwu (%)'}, inplace=True)

    return df

# 示例用法
if __name__ == "__main__":
    file_path = os.environ["TEST_OUTPUT_PATH"]  # NCCL 测试输出文件路径
    peak_bw = float(os.environ["PEAK_BANDWIDTH"])  # 峰值带宽 (GB/s)
    df = process_nccl_output(file_path, peak_bw)
    print(df)
    with open(os.environ["TEST_REPORT_PATH"], "w") as f:
        df.to_csv(f, index=False)