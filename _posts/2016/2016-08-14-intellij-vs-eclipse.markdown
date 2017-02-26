---
published: true
title: Intellij vs Eclipse
layout: post
---
Intellij实在是名气太大，尝试数次，记录一下。

## 界面

初次安装Intellij 2016.2.1，很不适应的就是Intellij的字体渲染，在我的1366的笔记本还是1920的显示器上，编辑器里的代码都没法看，非常的虚，好在可以设置字体：

File > settings > Editor > Colors&Fonts > Font 下把Primary font修改为 Consolas，一切都ok了。

窗口布局方面，我喜欢Itellij的统一方式，编辑和调试，都是同一个布局，不需要像Eclipse切来切去。

## 编辑能力

Eclipse的编辑能力一般，尤其有个不方便的地方就是很多文件，可能都会是shell打开文件，而不是内置编辑，Intellij则内置了对大部分文件的高亮和编辑功能，比如css/js/html都可以不再借助外部编辑器了。所以从编辑能力来看，Inteliij是大胜Eclipse的，唯一让我想念的是Eclipse内置的WindowsBuilder对GUI的编辑。

## 集成的插件

和标准Eclipse版本相比，Intellij集成了更多的插件和工具，比如Svn、Git、Maven等，这些很方便，当然我也担心过多的插件会影响IDE的性能和稳定性，不过Intellij在这方面似乎还不错。

## hotswap

这是Eclipse非常好用的一个功能。Intellij也实现，Eclipse里每次按Ctrl-S保存时，就会自动编译并作hotswap，IntelliJ里不需要Ctrl-S，但需要手动触发编译，按Ctrl-F9，这个键比Ctrl-S稍微难按一点，但和Eclipse的效果基本一样，如果是在外部修改文件，Eclipse需要按右键，Refresh工程，Itellij还是按Ctrl-F9。

## 快捷键
Intellij号称快捷键很方便，但我发现一个问题是很多的快捷键在远程桌面里没法按（远程桌面屏蔽了一部分的快捷键组合），这使得透过远程桌面使用Inteliij变得不方便了。 Itellij下常用的快捷键：

- Ctrl-Alt-left  回退
- Shft-F6 Refactor的改名
- Ctrl-F3 当前选中的搜索，类似Eclipse Ctrl-K
