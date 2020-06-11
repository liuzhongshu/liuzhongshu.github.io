---
published: true
title: Autohotkey
layout: post
---

autohotkey是一个windows快捷键神器，可以做很多事情。

## 安装
使用zip解压就可以了，然后再解压的目录下创建一个和exe同名的ahk文件，编辑好，双击exe就可以了。

## 指定窗口总在最前

用win+p来指定一个窗口在最前，再按一次取消。

```
#p::  Winset, Alwaysontop, , A
```