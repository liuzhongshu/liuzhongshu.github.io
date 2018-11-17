---
published: true
title: Cordova vs React Native
layout: post
---

这两个是移动端跨平台开发当前最流行的技术方向，做下比较：

## 控件

这是两者最大的区别，Cordova原生没有什么控件，但是有大量的第三方框架可以提供，比如Quasar，ReactNative自身带了少量的控件，比较少，也有不少第三方框架可以用。从数量上看，两者都挺丰富，Cordova毕竟有着Web这么多年的积累，相对更丰富一些。

但从控件的实现上，二者有了明显的差别，Cordova最大一个缺点，也是ReactNative最大的优点，就是是否支持原生控件。

## 插件


## 调试
Cordova+Webpack对hot reload的支持非常非常好，几乎大部分的修改都可以hot reload，速度也很快，ReactNative虽然支持hot reload，但是很多时候不能正常工作，而且ReactNative需要手工打开hot reload，这点也不如Cordova方便。

在连接chrome调试时，Cordova项目和调试一个本地Web应用没什么差别，ReactNative也不错，但是不能调试界面元素（inspect），这点不如Cordova。

Webpack在调试设置断点上，和chrome配合有明显的[问题](https://github.com/webpack/webpack/issues/3165)，通常需要切换devtool到一个合适的模式，我的环境中，使用inline-source-map要好些，但性能较差，而且会导致chrome远程调试不可用(自动断开)，所以可以根据情况来选择。

另一点值得注意，就是cordova平台大多可以通过桌面浏览器来调试，脱离手机，这在某些时候也是很方便的，RN没有这个支持。

## 权限

因为Webview对权限的要求很低，所以一般使用Cordova开发，基本上要的权限也很少，在安装APP的时候比ReactNative看起来要好一些。

## 总结

| | ReactNative | Cordova
- | :-: | -:
二进制大小 | 至少8M | 非常小(K级别) 
控件 | 原生或js实现 | Web控件
样式 | 简化CSS | CSS
编译 | 慢 | 较快
桌面调试 | 不支持 | 支持
HotReload | 基本无用 | 好用

从工程角度，大部分方面Cordova+webpack要更稳定易用一些，UI的性能ReactNative要快一些。
