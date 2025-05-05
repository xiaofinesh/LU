import pandas as pd

# 加载数据
report_data = pd.read_csv("table2.csv", usecols=[0, 2], encoding="gb18030")
results_data = pd.read_csv("table1.csv", encoding="gb18030", usecols=[0, 1, 2, 3, 4, 5, 6])
results_data.columns = ['序号', '腔体编号', '最高值', '最低值', '压强差值', 'TRUE/FALSE', '时间']

# 数据预处理
report_data = report_data.rename(columns={report_data.columns[0]: "时间", report_data.columns[1]: "压强"})
report_data['压强'] = pd.to_numeric(report_data['压强'], errors='coerce').fillna(0)
report_data['时间'] = pd.to_datetime(report_data['时间'], errors='coerce')
report_data = report_data.dropna(subset=['时间'])

results_data['时间'] = pd.to_datetime(results_data['时间'], errors='coerce')
results_data = results_data.dropna(subset=['时间'])

# 初始化新表
new_data = []

# 遍历 results_data 的每一行
for _, row in results_data.iterrows():
    if row['腔体编号'] != 1:  # 只处理腔体编号为1的行
        continue

    result_time = row['时间']
    max_pressure = row['最高值']
    min_pressure = row['最低值']
    true_false = row['TRUE/FALSE']
    pressure_diff = row['压强差值']

    # 放宽时间范围
    time_range_data = report_data[
        (report_data['时间'] >= result_time - pd.Timedelta(seconds=15)) &
        (report_data['时间'] <= result_time - pd.Timedelta(seconds=5))
    ]

    # 改进压强匹配（寻找最接近的压强值）
    time_range_data['压强差'] = (time_range_data['压强'] - max_pressure).abs()
    matching_pressure_row = time_range_data.nsmallest(1, '压强差')

    if not matching_pressure_row.empty:
        matching_time = matching_pressure_row.iloc[0]['时间']
        matching_pressure = matching_pressure_row.iloc[0]['压强']

        # 获取后续39个压强值
        subsequent_pressures = report_data[
            report_data['时间'] > matching_time
        ].head(39)['压强'].tolist()

        # 如果不足39个值，用0填充
        while len(subsequent_pressures) < 39:
            subsequent_pressures.append(0)

        # 构造新行
        new_row = [
            result_time, max_pressure, min_pressure, true_false, pressure_diff, matching_pressure
        ] + subsequent_pressures

        new_data.append(new_row)

# 定义新表的列名
columns = ['时间', '最高值', '最低值', 'TRUE/FALSE', '压强差值', '匹配压强值'] + [f'压强值_{i}' for i in range(1, 40)]

# 创建 DataFrame 并保存为 CSV
new_df = pd.DataFrame(new_data, columns=columns)
new_df.to_csv("Processed_Data1.csv", index=False, encoding="gb18030")

print("数据处理完成，新文件已保存为 Processed_Data1.csv。")