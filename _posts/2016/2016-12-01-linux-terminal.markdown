---
published: true
title: Linux命令行
layout: post
---
Linux命令行(termial)可以做很多有趣的事情，以Ubuntu为例做个记录。

## mosh

这是我通常在VPS里第一个安装的软件，以提供‘更可靠’的远程连接，通常这样就可以了。

```
sudo apt-get install mosh
```

遗憾的是mosh在各大发行版仓库里的最新版本(1.2.4)有个问题，对于鼠标支持不好，如果有鼠标的需求（比如下面的w3m），就需要安装最新的mosh，这样做：

```
sudo add-apt-repository ppa:keithw/mosh-dev
sudo apt-get update
sudo apt-get install mosh
```

服务器端安装好之后，客户端也同样安装一下，然后不需要启动任何服务，就可以和平时使用ssh一样使用了。Windows下也可以使用MobaXterm做客户端，自带mosh。

## w3m

w3m是命令行下的浏览器，比Lynx好用的地方在于支持鼠标，对的，Terminal下的鼠标操作，点击和滚动都支持，有时候（你懂的）值得一用，安装就apt-get就好了。常用的键盘操作：

- 空格：向下翻屏
- b: 往前翻屏
- 方向键：移动光标
- TAB：下一个链接
- ESC+TAB：前一个链接
- 回车：编辑文本或打开链接
- B：返回前页
- q: 退出

注意：如果想支持鼠标，对终端有要求，Windows下的Putty是可以支持的，除此之外还没有发现其他终端可以支持鼠标。

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

## Gmail

要访问gmail，可以用命令行的邮件客户端alpine, 用apt-get安装即可，参考这个[帖子](http://askubuntu.com/questions/130899/how-can-i-configure-alpine-to-read-my-gmail-in-ubuntu)设置，很好用，理论上也可以设置多个gmail，或者使用别的邮箱也可以。

这样设置好之后，收发邮件都没有问题。

## Screen

这个是终端下的又一神器，可以在一个终端下多任务，切换多个窗口，apt-get安装，执行screen启动，然后就可以放行的执行各种任务了，也不用担心ssh断开后任务会终止，下次重连后，用screen -ls可以列出还在运行的session，用screen -R 就可以恢复最近一个session，如果你又几个窗口开了putty，可以用screen -x复用到一个session上。

更棒的是，可以随时切换session，下面是几个常用快捷键：

* ctrl-a c：创建一个新的 session
* ctrl-a ctrl-a：在 session 间切换
* ctrl-a n：切换到下一个 session
* ctrl-a p：切换到上一个 session
* ctrl-a 0…9：同样是切换各个 session
* ctrl-a d：退出 Screen

如果想在screen里也支持鼠标，需要启动screen时指定终端类型：

```
screen -T xterm
```

## NCDU

控制台下没有好用的文件管理器，有时管理文件就比较麻烦，还好有一个神器 NCDU，可以用 apt-get 安装，然后执行 ncdu / 就知道有多好用了。
