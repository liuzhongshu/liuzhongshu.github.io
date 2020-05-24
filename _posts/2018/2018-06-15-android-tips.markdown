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

## 实时投射和远程控制

可以安装scrcpy，这个是命令行工具，而且同时很好的解决了手机输入困难的问题。

## 剪贴板访问

安卓10下，后台应用无法再直接读取剪贴板内容了，虽说是个隐私保护，但却导致所有剪贴板同步工具不能使用了，非常遗憾，如果能做成权限多好。好在可以通过[adb方式](https://www.webplover.com/android-10-clipboard-solution/)赋权。


