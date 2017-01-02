---
published: true
title: Windows下的VPN
layout: post
---
Windows下自带的VPN主要是PPTP和L2TP两种，两种都用过，Server端用Windows 2008R2，客户端各种Windows都可以。总的来说还是方便的，毕竟都内置，不用再安装软件。

## 获取

```
git clone git@github.com:liuzhongshu/liuzhongshu.github.com.git
```

## 本地操作

- 添加所有修改，包括删除的文件```git add . -A```
- 查看修改情况```git status```
- 提交到本地```git commit -m "comment"```
- 取消，下面两条都可以，暂存区域会取消，--hard会将修改的文件恢复到上次commit状态。

```
git reset
git reset --hard
```

- 比较，下面三条依次用最新版本，上次版本，上上次版本和本地进行比较

```
git diff 
git diff HEAD^
git diff HEAD~2
```

## 远程服务器

添加远程服务器```git remote add origin url```

## 其它
