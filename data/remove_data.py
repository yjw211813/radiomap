import os
import shutil
import sys


def delete_all_in_directory(directory):
    """
    删除指定目录下的所有文件和子目录
    """
    if not os.path.isdir(directory):
        print(f"错误：'{directory}' 不是一个有效的目录路径")
        return False

    try:
        # 遍历目录中的所有条目
        for entry in os.listdir(directory):
            full_path = os.path.join(directory, entry)

            if os.path.isfile(full_path) or os.path.islink(full_path):
                # 如果是文件或符号链接，直接删除
                os.unlink(full_path)
                print(f"已删除文件: {full_path}")
            elif os.path.isdir(full_path):
                # 如果是目录，递归删除
                shutil.rmtree(full_path)
                print(f"已删除目录: {full_path}")

        print(f"已清空目录: {directory}")
        return True
    except Exception as e:
        print(f"删除过程中发生错误: {e}")
        return False


if __name__ == "__main__":


    target_dir = r"C:\Users\Administrator\Desktop\TS_communication_py\git_proj\radio_map_construction\data\RadioMapSeer"

    # 获取绝对路径
    target_dir = os.path.abspath(target_dir)

    # 确认操作
    print(f"警告: 这将删除目录 '{target_dir}' 下的所有文件和子目录!")
    confirmation = input("确定要继续吗? (输入 'yes' 继续): ")

    if confirmation.lower() == 'yes':
        delete_all_in_directory(target_dir)
    else:
        print("操作已取消")