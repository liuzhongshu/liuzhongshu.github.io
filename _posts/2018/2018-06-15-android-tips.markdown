---
published: true
title: android提示
layout: post
---

## 录制屏幕

有android开发环境的话，录制手机视频就很简单，连上usb线，执行下面这条
```
adb shell screenrecord --verbose ./sdcard/screencast-video.mp4
```
后面的文件路径是手机端的，所以要放在sdcard目录下，如果录制的时候想看到点击，可以在设置里先打开“show touches”。

## 实时投射

可以安装一个chrome app，叫做Vysor，然后usb链接手机，就可以把手机屏幕投射到电脑上。

* 可以在电脑上控制手机，输入英文也可以，中文用电脑端的中文输入法是不行的，但是可以在手机上切换到中文，再用PC键盘输入是可以的（PC切换到英文）。
* 剪贴板从手机到电脑可以，反之不通。

更好的方法是scrcpy，这个是命令行工具，adb建立连接后，就可以用scrcpy实时投射，用起来和Vysor一样，可以通过命令行数控制分辨率(-m)和带宽(-b)，比如

```
scrcpy -m 700 -b 1m
```

