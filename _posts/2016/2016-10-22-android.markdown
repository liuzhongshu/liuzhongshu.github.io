---
published: true
title: Android开发环境
layout: post
---
## Android Studio

从[这里](http://www.androiddevtools.cn/)下载Android Studio，下载最大最新的那个，含sdk和各种tools，省得后续下载的痛苦。

安装的时候，可以选择安装路径，如果不按缺省路径，安装完成后记得设置环境变量ANDROID_HOME到sdk的目录。因为后续很多步骤是依赖这个变量来找sdk。

sdk的目录下有个SDK Manager.exe，这个可以用来继续安装更多的sdk，默认的话，一般只带一个比较新的sdk，可以多安装几个sdk，也可以等到需要再装。

## 国情设置

在启动Android Studio之前，针对我们国情，建议增加`c:\users\xxx\.gradle\init.gradle`文件，内容如下：

```
allprojects {
    repositories {
        maven{ url 'http://maven.aliyun.com/nexus/content/groups/public'}
    }
}
```

这个为gradle设置一个镜像，可以防止gradle从中央库下载文件（非常慢）。

## 运行studio

第一次运行，会有的一个向导，可以取消掉，没有关系。但是接下来，一定不要open之前的项目，因为项目需要的gradle版本不一致，一样可能会卡死，所以保险的方式是先创建一个HelloWorld，成功之后把其下gradle\wrapper\gradle-wrapper.proerties文件的最后一行拷贝出来，以后所有的项目都用这一行，类似下面这个：

```
distributionUrl=https\://services.gradle.org/distributions/gradle-2.14.1-all.zip
```

也就是说，所有的项目都要修改这个proerties文件，确保用同一个gradle版本，以免去下载，理论上这样可能有些问题（万一gradle不兼容），所以另一个解决下载gradle卡死的方法是先让他卡，然后杀掉Android Studio，再去 [这里](http://services.gradle.org/distributions/) 下载对应的版本，然后放到

```
C:\Users\用户名\.gradle\wrapper\dists\gradle-xxx-all\sdfdfa4w5fzrksdfaweeut
```

后面这串乱码是Android Studio在打开工程时自动建出来的，所以要让他先卡一下，把这个目录建出来（有点奇葩的设计），然后就可以放进去了。


## 模拟器

在Android Studio内可以管理模拟器，一定记得使能Intel Haxm，可以大大加速模拟器的速度，使能Intel Haxm会自动从intel下载一个驱动并自动安装，但是要在下次重启计算机之后，Haxm才会起到加速效果，所以如果安装了Haxm，还是很慢的话，可能需要重启下。

## 真机调试

真机调试需要在手机上开USB调试选项，然后用USB接到开发机上，安装驱动，就可以调试了，不过不是每个手机都容易安装驱动，有些手机非常不容易安装驱动，如果实在搞不定，参考[这个帖子](http://www.makeuseof.com/tag/android-wont-connect-windows-adb-fix-it-three-steps/) 一步一步安装应该可以成功。

** 批评：小米手机就是非常不友好的典型，不推荐使用 **

有的时候，换个手机还是不能识别，可以从设备管理器里手动安装驱动，选择上面安装好的驱动程序，就可以了，参考下图：

![](../../public/images/2017-12-02-15-49-37.png)

也可以使用无线调试，需要手机和开发机在同一局域网内，推荐在Android Studio下安装一个插件ADB Wifi, 

然后在USB连上手机的情况下，选择 Tools/Android/ADB Wifi/ADB USB to Wifi，就建立了无线的ADB链接了，以后断开之后，需要在Android Studio的命令行下输入

```
adb connect 192.168.xx.xx
```

可以重新建立无线连接（再也不需要USB线了），如果找不到adb，先进入sdk下的platfrom-tools这个目录再执行。

运行的时候如果出This version of android studio is incompatible with the gradle version used.Try disabling the instant run，可以在 Settings/Preferences > Build, Execution, Deployment option > Instant Run 下面禁用Instance run所有选项即可。

## logcat

logcat用来看android日志，不过这个命令设计的不那么友好，常见的使用场景都要加很多参数，不琢磨一会，用起来比较麻烦，下面是两个例子：

导出日志到文件(-v time是为了加时间，-d是为了导出后退出，否则他一直监听日志就退不出了)：

```
adb logcat -v time -d > logcat.txt 
```

要想过滤某个进程的日志，logcat并不支持，logcat只支持按tag（就是log命令的第一个参数）和level过滤，显然按tag过滤很多时候不合适，多个进程可能有同名的tag，所以只能再加grep来按pid过滤，用 adb shell ps，找到进程pid，再用下面的命令来过滤：

```
adb logcat -v time | grep pid
```

这样的话，windows下只能用findstr了，另一个常见的问题是，logcat的内容太多了，有时候某些进程不停的写log，导致log很快满了而冲掉有用的log，但是android没有提供一个方法可以禁止某个进程记录log（除了杀死进程），有的时候（定位一些偶发问题）很不方便。

总之logcat的设计有点反人性。

如果设备不再手边，或没有条件使用USB线，可以在设备上安装一个[catlog](https://play.google.com/store/apps/details?id=com.nolanlawson.logcat&hl=en)工具，可以直接显示或保存日志，挺好用，但是这个工具需要root权限。

## 开发与亮屏

开发调试时，通常需要不停查看屏幕，默认的安卓设置会导致一会就熄屏了，虽然可以修改设置来禁止熄屏，但是改来改去还是蛮麻烦的，简单的方法当然是安装一个软件，这里软件很多，我用的是Stay Alive，还不错。

## 多语言
如果开发多语言APP，想测试英文版本，需要修改测试手机的locale，可以安装小软件来做，比如MoreLocale 2，切换的时候，需要ROOT权限，或者也可以通过adb赋给权限，这样做：
```
adb shell pm grant jp.co.c_lis.ccl.morelocale android.permission.CHANGE_CONFIGURATION
```
修改后，app都不需要重启，权限即刻生效。

应用程序想要支持多语，自然是要在应用里做，比如Vue可以用vue-i18n，但是如果想要应用程序的名称、权限根据语言来定，就需要在cordova上安装这个[插件](https://github.com/kelvinhokk/cordova-plugin-localization-strings),很好用。
