---
published: true
title: iOS开发环境
layout: post
---
## 虚拟机安装

使用  Virtual Box 5.x，可以虚拟最新的OSX 10.11，没有问题，详细方法参考[这里](https://techsviewer.com/how-to-install-mac-os-x-el-capitan-on-pc-on-virtualbox/), 需要注意的是细节是：

- 虚拟机内存要尽量大，CPU核数尽量多，VideoRam设置到最大（128M）
- 取消软驱，芯片组使用PIIX3

然后需要**关闭VirtualBox**, 修改以下参数：

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

VBoxManage setextradata "OSX10.11" "VBoxInternal/Devices/smc/0/Config/GetKeyFromRealSMC" 1

后面的1就是分辨率，可以设置以下几个值：

- 0 : 640x480
- 1 : 800x600
- 2 : 1024x768
- 3 : 1280x1024
- 4 : 1440x900
- 5 : 1900x1200

## Xcode

直接通过App Store安装就可以，Xcode8支持简化的Sign方法，参考官方文档。基本上，你可以不再需要注册Developer，使用普通的Apple ID就可以开发了。

## 真机连接

使用真机调试的前提是USB可以通，连接iOS设备后，可以在设备 》 USB 》 设置USB里配置过滤器选中iOS设备即可，很好用。