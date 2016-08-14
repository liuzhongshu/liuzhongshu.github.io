---
published: true
title: Intellij vs Eclipse
layout: post
---
Intellij实在是名气太大，尝试数次，记录一下。

##界面

初次安装Intellij 2016.2.1，很不适应的就是Intellij的字体渲染，在我的1366的笔记本还是1920的显示器上，编辑器里的代码都没法看，非常的虚，好在可以设置字体：

File > settings > Editor > Colors&Fonts > Font 下把Primary font修改为 Consolas，一切都ok了。

##编辑能力

Eclipse有个不方便的地方就是每次双击文件，可能都会是shell打开文件，而不是编辑文件，Intellij没有这个不便，并且内置了对大部分文件的高亮和编辑功能，比如css/js/html都可以不再借助外部编辑器了。

##集成的插件

和标准Eclipse版本相比，Intellij集成了更多的插件和工具，比如Svn、Git、Maven等，这些很方便，当然我也担心过多的插件会影响IDE的性能和稳定性，不过Intellij在这方面似乎还不错。

## hotswap

这是Eclipse非常好用的一个功能。Intellij不仅实现，还有所增强，在Eclipse里除了修改函数内部代码，其他修改都不能hotswap。