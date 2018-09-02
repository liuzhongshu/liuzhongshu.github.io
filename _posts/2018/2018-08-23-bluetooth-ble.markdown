---
published: true
title: 蓝牙BLE设备
layout: post
---

BLE是蓝牙低功耗的缩写。

## 角色

可以容易理解，central是控制，peripheral是外设，手环一类的设备是peripheral，手机通常是central，不过手机也可以做peripheral。

* 一个central可以连多个peripheral
* 在蓝牙4.0里，一个peripheral只能连一个central，但[4.1规范里可以一对多](https://stackoverflow.com/questions/29552626/can-a-peripheral-can-be-connected-to-multiple-centrals)
* peripheral会广播(advertising)信号出去，central可以scan到这些设备，并发起连接，peripheral不能主动发起连接
* 并且没有在advertising的设备无法被连接。所以要让peripheral连多个central，一定要在连接建立后继续发送advertis。

## central

central的流程
```
init => scan => connect => read/write/notify => disconnect
```
scan比较费电，所以应该在connect到设备后，停止scan。

## peripheral

peripheral的流程比较简单，手机当peripheral也是可以的，这样两部手机可以建立连接，手机做peripheral的处理过程如下：

```
init=> advertise => callback
```
这个init比central要复杂一些，因为需要注册一些想要实现的service。

advertise之后peripheral就不用主动做什么事情，等待连接并处理各种回调即可。但要注意对advertise的处理，如果在连接建立之后不关闭advertise，就可以做到连接多个central。

需要注意：作为peripheral设备的手机暴露的Mac地址[从Android 5开始不再是真实Mac](
https://stackoverflow.com/questions/36180407/why-the-address-of-my-bluetoothdevice-changes-every-time-i-relaunch-the-app)。每次开始advertising都会换一个虚拟的Mac地址，这导致一个问题就是如果peripheral每次断连之后重新advertise，central设备没办法直接reconnect（因为Mac变了）。

实际测试下来，有些手机是在每次开始广播的时候换Mac，有些手机则是定时更换，似乎是芯片有关，并且Android不提供API禁止这个特性。

## 数据处理

基于Gatt协议。

## 规范

Gatt是有规范的，常见的一些service在[这里](https://www.bluetooth.com/specifications/gatt/services)可以查到，比如电量信息就是0x180F。

## 工具
* nrf Connect 做调试工具还不错
