---
published: true
title: 网络唤醒
layout: post
typora-copy-images-to: ..\public\images\2017
---


在Xp上需要设置3个地方：

* BIOS里打开WOL，不同版本的BIOS，这个设置名称不同

* 网卡的WOL设置

![](../public/images/2017/lan-wol-1.png)

有意思的是，下面一个选项“唤醒功能”可以设置多种唤醒方式，缺省的方式会导致很多误唤醒，甚至关机后自动唤醒，我将这个选项设置为“幻数据包”就解决了。

* 网卡的电源管理

![](../public/images/2017/lan-wol-2.png)

设置好之后，可以用PC或者手机做唤醒，PC侧可以用[这个小工具](http://www.nirsoft.net/utils/wake_on_lan.html)，手机侧可以用Wake On Lan，我都成功了。

在Windows8.1，我的笔记本上，上述的最后一个设置有所不同，非常费解。
