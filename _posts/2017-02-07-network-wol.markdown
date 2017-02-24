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

* 网卡的电源管理

![](../public/images/2017/lan-wol-2.png)

设置好之后，可以用PC或者手机做唤醒，PC侧可以用[这个小工具](http://www.nirsoft.net/utils/wake_on_lan.html)，手机侧可以用Wake On Lan，我都成功了。

在Windows8.1，我的笔记本上，上述的最后一个设置有所不同，非常费解。
