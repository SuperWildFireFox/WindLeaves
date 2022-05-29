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
 python test_play_offline_by_human.py
```



## 在线AI

（如果B站首页小游戏还是风叶穿行的话）

首次运行请在控制台输入如下指令生成证书文件：

```
mitmproxy
```

转到 C:\Users\你的用户名\\.mitmproxy 文件夹，双击mitmproxy-ca-cert.p12，全部选择默认选项，安装证书。

之后，请在仓库根目录下打开控制台，输入指令：

```
mitmdump -s httpproxy.py -p 9090
```



