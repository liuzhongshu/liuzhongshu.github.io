---
published: true
title: Linux命令行
layout: post
---
Linux命令行(termial)可以做很多有趣的事情，以Ubuntu为例做个记录

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

服务器端安装好之后，客户端也同样安装一下，然后不需要启动任何服务，就可以和平时使用ssh一样使用了。Windows下也可以使用MobaXterm做客户端，自带mosh。

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

注意：如果想支持鼠标，对终端有要求，Windows下的Putty是可以支持的，除此之外还没有发现其他终端可以。

## googler

如果常用google搜索，可以安装这个，不过一般情况下用w3m就可以了。

```
wget https://github.com/jarun/googler/releases/download/v2.8/googler_2.8-1_all.deb
sudo dpkg -i googler_2.8-1_all.deb
```

这个的用法很简单

```
googler -n 5 keyword
````

如果想让googler和w3m集成，也很简单：

```
export BROWSER=w3m
```

这样用googler查询的结果，就可以用w3m打开浏览了。
