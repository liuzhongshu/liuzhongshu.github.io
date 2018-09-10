---
published: true
title: OSX调整
layout: post
---
新安装了OSX，记录以下调整:

### 安装brew

brew就是osx下的最好用的命令行包管理器，重装系统后，我第一件事通常就是安装brew，只需执行[官网](http://brew.sh)上的一行脚本，并且安装brew之后，xcode command line、git等开发工具也就安装好了。然后把cask安装上，用于安装gui程序：

```
brew tap caskroom/cask
```

brew每次install任何东西都会先update，挺烦的，可以增加一个环境变量禁止掉 export HOMEBREW_NO_AUTO_UPDATE=1


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

在Apple Script中写下面的脚本，并保存为app放在下载目录下，就可以一次清空下载目录了（脚本很酷不是）：

```
tell application "Finder" to move (every item of (container of item (path to me) as alias) whose name is not (name of item (path to me) as text)) to trash
```

### 系统字典
OSX下有系统字典，好处不言而喻，稳定且可以在任意程序中使用Option+Ctrl+D取词，添加词库方法如下：

- 下载[DictUnifier](http://code.google.com/p/mac-dictionary-kit/)
- 找一些词库，比如stardict的词库文件(tar.bz2)
- 运行DictUnifier，将词库拖放过去，开始转换词库。

转换过程比较慢，耐心等候就好，一个发现是DictUnifier我只能使用2.0，最新的2.1反而会转换失败，估计可能和我是osx10.6有关系。

### 其它

禁止spotlight索引```sudo mdutil -a -i off```