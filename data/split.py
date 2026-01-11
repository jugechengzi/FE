import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="处理并复制 JSON 数据集")
    parser.add_argument("--file", type=str, required=True, help="文件名 (例如: multi_counterfact_20877.json)")
    parser.add_argument("--skip", type=int, default=0, help="要跳过的前 N 条数据数量 (默认 0)")
    args = parser.parse_args()

    # 配置基础路径
    base_dir = "/home/svu/hkliu/model_edit/me/data"
    backup_dir = os.path.join(base_dir, "backup")
    
    src_path = os.path.join(backup_dir, args.file)
    dest_path = os.path.join(base_dir, args.file)

    if not os.path.exists(src_path):
        print(f"错误: 找不到源文件 {src_path}")
        return

    print(f"读取: {src_path}")
    with open(src_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_len = len(data)
    # 执行切片
    processed_data = data[args.skip:]
    final_len = len(processed_data)

    print(f"原始长度: {original_len} | 跳过: {args.skip} | 剩余: {final_len}")

    with open(dest_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
    print(f"成功保存至: {dest_path}")

if __name__ == "__main__":
    main()