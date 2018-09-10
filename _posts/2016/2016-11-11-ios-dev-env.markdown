---
published: true
title: iOS开发环境
layout: post
---
## 虚拟机安装

使用  Virtual Box 5.x，可以虚拟最新的OSX 10.11，没有问题，详细方法参考[这里](https://techsviewer.com/how-to-install-mac-os-x-el-capitan-on-pc-on-virtualbox/), 需要注意的是细节是：

- 虚拟机内存要尽量大，CPU核数尽量多，VideoRam设置到最大（128M）
- 取消软驱，芯片组使用PIIX3

然后需要**关闭VirtualBox**, 并修改以下参数：

```
cd "C:\Program Files\Oracle\VirtualBox\"
VBoxManage.exe modifyvm "OSX10.11" --cpuidset 00000001 000106e5 00100800 0098e3fd bfebfbff
VBoxManage setextradata "OSX10.11" "VBoxInternal/Devices/efi/0/Config/DmiSystemProduct" "iMac11,3"
VBoxManage setextradata "OSX10.11" "VBoxInternal/Devices/efi/0/Config/DmiSystemVersion" "1.0"
VBoxManage setextradata "OSX10.11" "VBoxInternal/Devices/efi/0/Config/DmiBoardProduct" "Iloveapple"
VBoxManage setextradata "OSX10.11" "VBoxInternal/Devices/smc/0/Config/DeviceKey" "ourhardworkbythesewordsguardedpleasedontsteal(c)AppleComputerInc"
```

然后可以启动了，安装完成后，进入OSX，可能会提示升级，在线升级即可。


## 分辨率

缺省的 分辨率是1024*768，这个如果要修改，可以参考[这里](http://www.wikigain.com/fix-macos-sierra-screen-resolution-virtualbox/)修改，简单的说，只有几种分辨率可以修改：

VBoxManage setextradata "OSX10.11" "VBoxInternal/EfiG"opMode" 4

后面的1就是分辨率，可以设置以下几个值：

- 0 : 640x480
- 1 : 800x600
- 2 : 1024x768
- 3 : 1280x1024
- 4 : 1440x900
- 5 : 1900x1200

如果没有对应你的显示器的分辨率，可以稍微设置大一些，把dock栏放左边，就可以全屏模式工作了。

## Xcode

直接通过App Store安装就可以，Xcode8支持简化的Sign方法，参考官方文档。基本上，你可以不再需要注册Developer，使用普通的Apple ID就可以开发了。App Store只能下载最新的xcode版本，如果需要老版本，可以在[苹果开发者](https://developer.apple.com/download/more/)选择下载，下载前先参考这张[表格](https://en.wikipedia.org/wiki/Xcode#8.x_series)，不要下载下来因为系统版本不够，无法安装。

8.0版本下载的格式多半是xip，这个格式需要系统至少是10.11.6，否则无法安装。

另外在双击安装之前，可以通过下面的命令去除冗长的校验(如果已经安装了，把下面的后一个参数改成/Application/Xcode.app)

```
xattr -d com.apple.quarantine Xcode_8.xip
```

如果不是做原生开发，而是cordova跨平台，也必须安装Xcode，否则也是编译不了的。

## 真机连接

使用真机调试的前提是USB可以通，连接iOS设备后，可以在设备 》 USB 》 设置USB里配置过滤器选中iOS设备即可，很好用。

如果使用xcode连接真机总是发生Lost Connection，需要做以下操作：
* 下载VirtualBox的Extension pack
* 在VirualBox的设置中安装下载的Extension pack
* 在虚拟机设置中，USB设置从1.0改为2.0

Xcode版本如果不够新，可能没有携带device support file，就不能连接高版本的iOS设备，重装Xcode太慢了，可以使用这个[项目](https://github.com/iGhibli/iOS-DeviceSupport)来手工安装device support file。

App下载到真机之后，会提示一个警告信任开发者的对话框，高版本的iOS没有提供“信任”的按钮，必须到设置》通用》设备管理中手工信任开发者，这样才可以运行。

## AD-HOC分发

为了测试，有时需要注册一些ios设备udid到开发者账号中，生成证书编译后，就可以在这些设备上测试，叫做AD-HOC分发，但是如果想通过网络安装(OTA)，参考[这里](https://stackoverflow.com/questions/23561370/download-and-install-an-ipa-from-url-on-ios), 整个过程还是比较复杂的，但有一个[网站](https://www.diawi.com/)可以大幅简化这个步骤。