---
published: true
title: Windows下的VPN
layout: post
---
Windows下自带的VPN主要是PPTP和L2TP两种，两种都用过，Server端用Windows 2008R2，客户端各种Windows都可以。总的来说还是方便的，毕竟都内置，不用再安装软件。

## 协议差别
PPTP基于TCP，L2TP基于UDP，这个基本差异导致了两者的兼容性有差别，比如在我这里，用移动宽带上网，PPTP是不通的，只能用L2TP。

## 服务器端
服务器端单网卡就可以，无需使用双网卡。

## 客户端
客户端如果用PPTP很容易，只需要建立基于PPTP的VPN即可，但是如果使用L2TP，则要麻烦一些，需要额外的配置注册表，在我的环境里，需要额外增加一项：
