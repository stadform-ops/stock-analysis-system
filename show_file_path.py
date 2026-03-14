import os


def list_root_files():
    root = os.getcwd()  # 当前运行目录（项目根目录）

    py_files = []
    csv_files = []

    for f in os.listdir(root):
        if os.path.isfile(os.path.join(root, f)):
            if f.endswith(".py"):
                py_files.append(f)
            elif f.endswith(".csv"):
                csv_files.append(f)

    print("PY FILES:")
    for f in py_files:
        print(f)

    print("\nCSV FILES:")
    for f in csv_files:
        print(f)


if __name__ == "__main__":
    list_root_files()