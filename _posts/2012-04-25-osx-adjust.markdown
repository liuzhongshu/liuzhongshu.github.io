---
published: true
title: OSX调整
layout: post
---
新安装了OSX，记录以下调整:

### 设置
- 触摸板允许click
- 屏幕四个顶点定义快捷功能，左上位“所有窗口”，右上为Space，左下为显示桌面，右下为关闭显示器。
- 调整Dock的位置和大小，我偏好将Dock放在左侧。


### shell
- 字体使用 Courier New 12
- 喜欢更简洁的提示符，在.profile中增加```export PS1="\W>"```
- 为了支持用上下键盘直接在命令历史中检索，在.inputrc中增加

```
"\e[A": history-search-backward
"\e[B": history-search-forward
```

bash在运行的时候分两种情况，login shell和非login shell。
login shell会执行下列文件：

* /etc/profile
* ~/.bash_profile
* ~/.bash_login(没有.bash_profile的情况下运行)
* ~/.profile(没有.bash_login的情况下运行) 

非login shell则是 ~/.bashrc, osx和Linux不一样的是，每次运行Terminal，都是login shell，所以osx下一个常见问题就是.bashrc的设置不起作用，所以，可以在bash_profile下增加这样一行:

```
[ -r ~/.bashrc ] && source ~/.bashrc
```

这样，.bashrc中的设置就在osx下也可以起作用了。顺便，如果切换shell的话，可以用 
```
sudo chsh -s /bin/zsh username
```
临时切换shell，可以用
```
exec bash
```

### 脚本

在Apple Script中些下面的脚本，并保存为app放在下载目录下，就可以一次清空下载目录了：

```
tell application "Finder" to move (every item of (container of item (path to me) as alias) whose name is not (name of item (path to me) as text)) to trash
```

### 其它

禁止spotlight索引```sudo mdutil -a -i off```