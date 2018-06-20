---
published: true
title: android技巧提示
layout: post
---

## 录制屏幕
有android开发环境的话，录制手机视频就很简单，连上usb线，执行下面这条
```
adb shell screenrecord --verbose ./sdcard/screencast-video.mp4
```
后面的文件路径是手机端的，所以要放在sdcard目录下，如果录制的时候想看到点击，可以在设置里先打开“show touches”。