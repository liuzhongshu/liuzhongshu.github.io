---
published: true
title: Oppo watch
layout: post
---


Oppo watch应该是这个时间点最强android手表了，记录一下(41mm款)

## 基本规格

* 高通2500
* 41mm/46mm

## 双模
Oppo watch支持双系统，简单理解就是智能模式和长续航(手环)模式，智能模式下可以直接切换到长续航模式，然后充电或重启可以回到智能模式。

## 智能模式
是标准的安卓8，不是wear os，通知栏下的几个功能：wifi、亮度、静音模式、勿扰模式、影院模式、飞行模式、水下模式。

![](../../public/images/2020-10-10-17-50-33.png)

有手表应用市场，app很少，只有20几款。

## 表盘

Oppo的表盘非常不错，并且大部分表盘上的小部件支持自定义，而且除了1x1的小部件，也支持多种更大的小部件，非常类似标准Android，唯一遗憾不开放给第三方开发者。

## 安装应用
可以和智能手机一样打开adb调试模式，支持安装apk，也可以使用wifi adb，但是因为手表上不能看ip地址，需要用`adb shell ifconfig`，设计良好的原生apk，基本都可以运行，但是注意：

* ActionBar不见了，应用种如果使用getActionBar会返回null，容易出空指针，并且因为actionBar消失了，所以toolbar和menu无法触发了。
* 没有办法返回(swipe需要修改代码才能支持)，只能按物理键回到主页
* 大部分应用选择白背景，但手表上黑背景更好一些(省电协调)
* 一些跳转url，比如startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(mUrl)) 会进入一个Unspport feature页面。
* SeekBarPreference支持的不好
* 多点触摸，放大缩小都可以操作。但可能是因为屏幕小，一些点击拖动操作较难触发。
* app可以保留后台（文档上说60秒后被杀），甚至似乎支持悬浮窗，android大部分功能得以保留。

解决方法：
* 增加swipeBack，但是swipeback不能在普通手机上增加，否则会导致后面setContentView时异常，所以需要有个区别，目前发现可行的方法是：

```
if (activity.getPackageManager().hasSystemFeature("android.hardware.type.watch"))
    activity.getWindow().requestFeature(Window.FEATURE_SWIPE_TO_DISMISS);
``` 

cordova应用：

* webview没有问题，但性能较慢。
* 标题栏没有问题了，因为cordova应用的标题栏都是在webview里的。
* 可以支持swipe back，因为是webview支持的。
* 播放音频没有问题。

可见，除了性能较慢，cordova应用几乎完美运行。

## 卸载应用
除了adb可以卸载应用外，可以通过UI来下载 设置 > 其他设置 > 应用管理

## 长续航模式

功能太少，只有心率，计步，通知和NFC公交几个功能，这个模式相较其他大号手环厂商就差多了，最起码闹钟都没有，对我就不够用了。

## 声音和震动

* 音量还可以，音质一般，可以后台播放。
* 震动只有两级可调，震感比较弱。

## 充电

充电快，即使只连接电脑USB接口，也可以大约1个小时充满。

## 截屏

没有音量键截屏，用scrcpy可以截屏。