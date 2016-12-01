---
published: true
title: Linux命令行
layout: post
---
Linux命令行(termial)

## mosh

这是我通常在VPS里第一个安装的软件，以提供‘更可靠’的远程连接，通常这样就可以了。

```
sudo apt-get install mosh
```

遗憾的是mosh在库的版本有个问题，对于鼠标支持不好，如果有鼠标的需求（比如下面的w3m），就需要安装最新的mosh，这样做：

```
sudo add-apt-repository ppa:keithw/mosh-dev
sudo apt-get update
sudo apt-get install mosh
```

## w3m

w3m是命令行下的浏览器，比Lynx好用的地方在于支持鼠标，对的，Terminal下的鼠标操作，有时候（你懂的）值得一用，安装就apt-get就好了。常用的键盘操作：

- 空格：向下翻屏
- b: 往前翻屏
- 方向键：移动光标
- TAB：下一个链接
- ESC+TAB：前一个链接
- 回车：编辑文本或打开链接
- B：返回前页
- q: 退出

## googler

如果常用google搜索，可以安装这个，不过一般情况下用w3m就可以了。

```
wget https://github.com/jarun/googler/releases/download/v2.8/googler_2.8-1_all.deb
sudo dpkg -i googler_2.8-1_all.deb
```
