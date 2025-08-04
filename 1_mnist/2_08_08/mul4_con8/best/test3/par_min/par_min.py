def find_max_train_epoch(accu_file_path):
    max_value = float('-inf')
    max_epoch = None

    with open(accu_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 略過表頭
            parts = line.strip().split()
            if len(parts) >= 3:
                epoch = int(parts[0])
                train = float(parts[1])
                if train > max_value:
                    max_value = train
                    max_epoch = epoch

    return max_epoch, max_value


def extract_epoch_block(input_file, output_file, target_epoch):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_lines = []
    write_flag = False

    for line in lines:
        if line.strip().startswith('epoch'):
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == str(target_epoch):
                write_flag = True
                output_lines.append(line)
            elif write_flag:
                break  # 遇到下一個 epoch，停止寫入
        elif write_flag:
            output_lines.append(line)

    with open(output_file, 'w') as f_out:
        f_out.writelines(output_lines)

    print(f"最大 train 值出現在 epoch {target_epoch}，該區塊已寫入 {output_file}")


# 路徑設定（請依實際情況修改）
accu_file_path = '../accu_sum'
data_file_path = '../par'
output_file_path = '../qiskit/par'

# 執行主流程
epoch, _ = find_max_train_epoch(accu_file_path)
extract_epoch_block(data_file_path, output_file_path, epoch)

