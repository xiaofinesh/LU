import pandas as pd

# 加载数据
report_data = pd.read_csv("table2.csv", usecols=[0] + list(range(2, 27)), encoding="gb18030")  # 加载表2，选择第1列时间和第2到第26列的腔体压强数据
results_data = pd.read_csv("table1.csv", encoding="gb18030", usecols=[0, 1, 2, 3, 4, 5, 6])  # 加载表1
results_data.columns = ['序号', '腔体编号', '最高值', '最低值', '压强差值', 'TRUE/FALSE', '时间']
results_data = results_data[~((results_data['最高值'] == 0) & (results_data['最低值'] == 0))]  # 过滤掉最高值和最低值均为0的行

# 数据预处理
report_data = report_data.rename(columns={report_data.columns[0]: "时间"})  # 重命名表2的时间列
report_data['时间'] = pd.to_datetime(report_data['时间'], errors='coerce')
report_data = report_data.dropna(subset=['时间'])  # 删除无效时间值

# 将表2的压强数据列按腔体编号命名
for i in range(1, 26):
    report_data[f"腔体{i}"] = report_data.iloc[:, i]  # 创建腔体编号的列

# 处理表1的数据
results_data['时间'] = pd.to_datetime(results_data['时间'], errors='coerce')
results_data = results_data.dropna(subset=['时间'])  # 删除无效时间值

# 初始化新表
new_data = []

# 遍历表1的每一行
for _, row in results_data.iterrows():
    result_time = row['时间']
    max_pressure = row['最高值']
    min_pressure = row['最低值']
    true_false = row['TRUE/FALSE']
    pressure_diff = row['压强差值']
    cavity_number = int(row['腔体编号'])

    # 放宽时间范围（30秒到10秒）
    time_range_data = report_data[
        (report_data['时间'] >= result_time - pd.Timedelta(seconds=12)) & 
        (report_data['时间'] <= result_time - pd.Timedelta(seconds=8))
    ]

    # 调试信息：输出匹配的时间范围
    print(f"腔体 {cavity_number} 的时间范围: 从 {result_time - pd.Timedelta(seconds=30)} 到 {result_time - pd.Timedelta(seconds=10)}")

    # 如果当前腔体编号在报告数据中存在压强列
    if f"腔体{cavity_number}" in time_range_data.columns:
        # 获取压强列并计算压强差
        time_range_data['压强差'] = (time_range_data[f"腔体{cavity_number}"] - max_pressure).abs()
        matching_pressure_row = time_range_data.nsmallest(1, '压强差')

        if not matching_pressure_row.empty:
            matching_time = matching_pressure_row.iloc[0]['时间']
            matching_pressure = matching_pressure_row.iloc[0][f"腔体{cavity_number}"]

            # 获取后续39个压强值
            subsequent_pressures = report_data[
                report_data['时间'] > matching_time
            ].head(39)[f"腔体{cavity_number}"].tolist()

            # 如果不足39个值，用0填充
            while len(subsequent_pressures) < 39:
                subsequent_pressures.append(0)

            # 构造新行，包括腔体编号
            new_row = [
                result_time, max_pressure, min_pressure, true_false, pressure_diff, matching_pressure, cavity_number
            ] + subsequent_pressures

            new_data.append(new_row)
        else:
            print(f"警告：腔体{cavity_number} 的时间范围没有匹配数据 (时间: {result_time})")
    else:
        print(f"警告：报告数据中没有腔体{cavity_number} 的压强列")

# 定义新表的列名
columns = ['时间', '最高值', '最低值', 'TRUE/FALSE', '压强差值', '匹配压强值', '腔体编号'] + [f'压强值_{i}' for i in range(1, 40)]

# 创建 DataFrame 并保存为 CSV
new_df = pd.DataFrame(new_data, columns=columns)
new_df.to_csv("Processed_Data1.csv", index=False, encoding="gb18030")

print("数据处理完成，新文件已保存为 Processed_Data1.csv。")
