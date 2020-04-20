---
published: true
title: android提示
layout: post
---

## 录制屏幕
很多手机自带录制屏幕，比如EMUI就有，就很简单了，如果录制的时候不想看到点击，可以在开发人员选项里先取消“show touches”。

如果没有自带工具，有android开发环境的话，录制手机视频也很简单，连上usb线，执行下面这条

```
adb shell screenrecord --verbose ./sdcard/screencast-video.mp4
```
后面的文件路径是手机端的，所以要放在sdcard目录下，如果录制的时候想看到点击，可以在设置里先打开“show touches”。

## 实时投射

可以安装scrcpy，这个是命令行工具，adb建立连接后，就可以用scrcpy实时投射，用起来和Vysor一样，可以通过命令行数控制分辨率(-m)和带宽(-b)，比如

```
scrcpy -m 700 -b 1m
```


