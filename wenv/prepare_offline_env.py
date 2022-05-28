import shutil
import zipfile
import os
import sys

ZIP_FILE_PATH = "offline_game/bczhc.github.io-master.zip"
SAVE_PATH = "offline_game/game"
root_dir = os.path.abspath(os.path.join(os.getcwd())).replace("\\","/")

# filename:[old_str,new_str]
REPLACE_MAP = {
    "index.html": ["/wind-game/", ""],
    "odXH9yzdsj.js": ["/wind-game/",
                      root_dir+"/offline_game/game/bczhc.github.io-master/wind-game/"]
}

# 删除旧文件
if os.path.isdir(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)

# 解压文件到game文件夹
with zipfile.ZipFile(ZIP_FILE_PATH, "r") as fp:
    for file in fp.namelist():
        if "wind-game/" in file:
            fp.extract(file, SAVE_PATH)  # 解压位置

# 找到对应文件，开始替换文字
for dir_path, sub_paths, files in os.walk(SAVE_PATH):
    for file in files:
        if file in REPLACE_MAP:
            file_full_path = os.path.join(dir_path, file)
            with open(file_full_path, "r", encoding="utf-8") as fp:
                content = fp.read()
                content = content.replace(REPLACE_MAP[file][0], REPLACE_MAP[file][1])
            with open(file_full_path, "w", encoding="utf-8") as fp:
                fp.write(content)

print("success!")
