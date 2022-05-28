# 用PPO算法玩B站风叶穿行游戏

## 程序说明

本程序基于windows系统，python3.7.9，pytorch1.7.1

## 准备本地游戏环境

感谢bczhc同学的风叶穿行离线版实现，请[下载](https://github.com/bczhc/bczhc.github.io/archive/refs/heads/master.zip)其仓库并将zip文件放置于offline_game文件夹

运行以下代码构造离线版环境：

```
python wenv/prepare_offline_env.py
```

如果想自己玩一下，可以输入：

```
 python wenv/test_play_offline_by_human.py
```

