---
published: true
title: 安卓设备作为开发机
layout: post
---

## 相关App
* F-Droid  - 相对安全的android市场
* 谷歌拼音输入法 - 对蓝牙键盘支持更好
* Chrome 
* Termux - Debian兼容的chroot环境

## Termux

好用的本地终端，并且具有包扩展能力。使用pkg或apt进行包安装，pkg只是apt的一个wrapper，可以简化apt的使用（比如不用再输入apt update了），用pkg list-all列举所有可安装的包，官方警告不要使用root安装apt包，很危险，可能会弄乱系统。

* 没有蓝牙键盘的话，可以安装一个Hacker Keyboard输入法，否则很多键打起来太麻烦。
* pkg upgrade 升级自带的包
* 安装后需要开启存储访问权限，然后termux-setup-storage，创建一些公共链接目录，让访问文件系统更便捷
* 常用的工具 pkg install wget vim git coreutils termux-api termux-tools grep tree ncurses-utils openssh gpg
* 大部分流行的语言Termux都可以支持，只需要 pkg install python cmake ruby nodejs golang, 
* 可以考虑安装tmux，复用session
* 在外部存储卡，为脚本文件增加执行权限(chmod 755)报错，可以将文件移到主目录，或者通过bash xxx.sh执行脚本就可以了。

Termux有一些很酷的用法，举例
* 通过其他app分享给termux，termux可以自动执行脚本，从而完成很酷的一些事情。
* 运行npm install -g web-code， 这个web-code可以通过浏览器编辑本地文件，一个超级简化的vscode

问题:
* android 8以上，npm会有一个报错，按照log文件的位置，修改maxConcurrentWorkers为1即可修复
* git clone 使用ssh url如果包unable to fork的错误，是因为没有安装ssh
* 没有蓝牙键盘的话，有些特殊字符不好输入，比如ctrl-c，可以按照一个hackerkeyboard，或者使用android原生输入法，android原生输入法中音量下键即为ctrl，上建alt。

## app_process
app_process 是Android 上的一个原生程序，可以从main()方法开始执行一个Java程序。scrspy就是利用这个来运行scrcpy-server进而得到shell权限。

```
app_process [java-options] cmd-dir start-class-name [options]
```

和windows下一样，通常我们需要先指定一个classpath，android可以将classpath指定为apk文件，比如
```
CLASSPATH=/data/local/tmp/scrcpy-server.apk app_process / com.genymobile.scrcpy.Server para1 para2
```

主要如果使用了proGuard，可能导致main函数被移除，所以应该再main上加上一个注解 @keep