---
published: true
title: Android开发环境
layout: post
---
## Android Studio

从[这里](http://www.androiddevtools.cn/)下载Android Studio，下载最大最新的那个，含sdk和各种tools，省得后续下载的痛苦。如果找不到完整包，只安装了studio，可以在上面网站的sdk-tools下安装android-sdk_rxxx-windows.zip，这个包下面会包含SDK Manager.exe，然后用这个exe去安装其余的sdk组件（不要用android studio的向导去安装sdk组件，没有这个单独的SDK Manager好用），无论用哪种方式安装了sdk，记得设置环境变量ANDROID_HOME到sdk的目录。因为后续很多步骤是依赖这个变量来找sdk。

使用SDK Manager安装sdk时，如果不用安卓虚拟机，可以不安装每个sdk下的各种system image，又大又不好用。

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

有的时候，换个手机还是不能识别，可以从设备管理器里手动安装驱动，选择上面安装好的驱动程序，就可以了，参考下图：

![](../../public/images/2017-12-02-15-49-37.png)


几个调试时遇到的问题：

* 运行的时候如果出This version of android studio is incompatible with the gradle version used.Try disabling the instant run，可以在 Settings/Preferences > Build, Execution, Deployment option > Instant Run 下面禁用Instance run所有选项即可。
* Android studio启动调试时出现unable to open debugger port, 重启Android studio也不管用，最后的解决方法是`adb kill-server`再`adb start-server`
* 有些手机每次调试安装应用的时候会弹出一个安装确认（比如华为的emui），这个可以在开发者选项中关闭。

## 无线连接adb

很多时候，usb连adb不是很方便，可以改用wifi，先在usb连接的情况下执行adb tcpip，这个命令把手机切换到无线adb模式，然后`adb connect 192.168.xx.xx`就可以了，只要不重启手机，每次都可以用这个adb connect连到手机。偶尔有时在wifi下会连接不成功，但使用手机做热点就没有问题。

如果手机已经root，上面使用usb的步骤可以忽略，通过app切换到wifi模式。

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

还有一种场景，应用crash了，因此只想看crash相关的日志(crash日志是一个单独的buffer），可以用

```
adb logcat --buffer=crash
```

我一般习惯在开发人员选项中把log的内存放到最大，log的级别有多个，有些手机（比如魅族）缺省会不记录debug级别的日志，可以在设置》开发者选项里调整。总之logcat的设计有点反人性，并且logcat并不persist，所以手机重启之后就没有了。

如果设备不再手边，或没有条件使用USB线，可以在设备上安装一个[catlog](https://play.google.com/store/apps/details?id=com.nolanlawson.logcat&hl=en)工具，可以直接显示或保存日志，挺好用，但是这个工具需要root权限。

## 签名冲突
apk的debug版本和release版本签名是不一致的，导致无法替换安装，会报错

```
signatures do not match the previously installed version; ignoring!
```

大部分时候下载之前的版本就可以了，可是有的时候，会发现根本这就找不到之前的版本，甚至通过adb shell pm list packages也找不到，那么可以强制“卸载”一次，再安装就可以了，这样：

```
adb unintall package-id
```
## app文件浏览
app可以使用内部存储或外部存储，内部存储仅对app开放，可以使用adb来方便浏览和检查，这样做：

```
adb shell
run-as your.package.id
ls -all
```

## 反编译

最好的工具组合是dex-tools和jadx，前者用于将dex转换为jar，后者阅读jar，虽然很多人推荐jd-gui来打开jar，但我发现jd-gui有些jar里的class文件打不开，而jadx可以。


## 开发与亮屏

开发调试时，通常需要不停查看屏幕，默认的安卓设置会导致一会就熄屏了，虽然可以修改设置来禁止熄屏，但是改来改去还是蛮麻烦的，简单的方法当然是安装一个软件，这里软件很多，我用的是Stay Alive，还不错。

## 多语言
如果开发多语言APP，想测试英文版本，需要修改测试手机的locale，可以安装小软件来做，比如MoreLocale 2，切换的时候，需要ROOT权限，或者也可以通过adb赋给权限，这样做：
```
adb shell pm grant jp.co.c_lis.ccl.morelocale android.permission.CHANGE_CONFIGURATION
```
修改后，app都不需要重启，权限即刻生效。

应用程序想要支持多语，自然是要在应用里做，比如Vue可以用vue-i18n，但是如果想要应用程序的名称、权限根据语言来定，就需要在cordova上安装这个[插件](https://github.com/kelvinhokk/cordova-plugin-localization-strings),很好用。

## google play

发布版本如果要上google play，过程如下：

* 注册并登录play console
* 添加app，在左侧标记的内容区依次填写（有些内容有先后，有些没有），基本信息区可以使用多语言（填写多份）
* 上传apk，填写分级信息，设置价格
* 如果应用有特殊的权限要求，必须填写隐私声明网址，这个比较麻烦
* 如果所有信息充分，play console会提示可以发布了，进入版本管理的地方，查看后就可以提交发布了。

提醒几点注意：

* 不要使用google 签名
* 如果你想app被全世界用户使用，最好缺省语言选英文
* 价格可以从收费转为免费，但不能反向操作

## apk优化

* 如果不需要appcompact这个库，可以在build.gradle中移除它，可以减小大约1M的体积

## Intent
Intent是安卓的一个核心概念，intent用于组件之间的调用和通讯，通常用于startActivity。intent有两种：

### Implicit intent
用于发起一个调用，但是不知道哪个应用会处理，也可能有多个应用可以处理，如果有多个，会让用户选择要给，典型的例子是分享，intent是这样的：

```
Intent i=new Intent();
i.setAction(Intent.ACTION_SEND);
```

### Explicit intent

这种intent指定了component，可以直接调用指定的app里的activity或service，比如：

```
Intent I = new Intent(getApplicationContext(),NextActivity.class);
I.putExtra(“value1” , “This value for Next Activity”);
```


## activity的filter属性

主Activity通常有这个属性：

```
<intent-filter>
     <action android:name="android.intent.action.MAIN" /> 
     <category android:name="android.intent.category.LAUNCHER" />
</intent-filter>
```

MAIN的意思表示是一个入口，通常和LAUNCHER一起出现，让桌面程序或第三方启动类程序可以发现这个activity。但是MAIN也可以和其他类别category组合比如和CATEGORY_CAR_DOCK组合则表示在dock时会执行的activity。

## mipmap和drawable

res下这两个目录都可以放图标。

## actionBar

早期安卓有一个ActionBar组件，后来安卓5.0增加了Toolbar，更灵活一些。

## service开发

service用于完成后台操作，它可以通过start或者bind启动，一个简单的service如下：

```
public class MyService extends Service {

  @Override
  public int onStartCommand(Intent intent, int flags, int startId) {
      return Service.START_NOT_STICKY;
  }

  @Override
  public IBinder onBind(Intent intent) {
    //TODO for communication return IBinder implementation
    return null;
  }
}
```

start一个service时，可以通过extra带参数，像这样：

```
Intent i= new Intent(context, MyService.class);
i.putExtra("KEY1", "Value to be used by the service");
context.startService(i);
```

这样会触发service里的onStartCommand方法，并且如果service还没有创建的话，service会被先创建起来，并调用onCreate方法，如果service已经创建了，就直接调用onStartCommand，这些调用都是通过UI thread进行，不会并发。

onStartCommand的返回值表示service是否需要restart，取值可以为：

* Service.START_STICKY  会restart，并且restart时传的Intent为null
* Service.START_REDELIVER_INTENT  会restart，并且restart时传上次的Intent
* Service.START_NOT_STICKY  不会restart

restart特性在不同的厂商间差别很大，很多厂商禁止了restart，比如小米、华为等，也就是说只要用户在多任务页面杀掉app，service也会被杀（即使标记了START_STICKY），解决方法似乎是加入白名单，参考[这里](https://stackoverflow.com/questions/41277671/clear-recent-apps-wipe-the-apps-memory-and-my-receiver-stopped-working/41360159#41360159)

执行stopService来停止service, 无论startService调用多少次，stopService一次就可以，或者也可以在service里，调用stopSelf停止自己。

Activity和service有几种通讯手段

* 通过startService带参数
* 通过bind
* 通过broadcasts和receivers，这中方法可以和非本进程的service通讯


### AccessibilityService

AccessibilityService比较特殊，体现在:

* AccessibilityService需要用户在系统设置里授权，授权后会由系统来启动，不过，手动startService再授权也可以，但如果没有授权，AccessibilityService的回调和操作不能生效。
* 如果是通过系统设置里授权启动，实际上只启动了Service，应用的Activity不会启动，也就是在多任务列表中看不到应用，但清理内存时还是会被清理掉。
* 清理内存或App强杀之后，无障碍授权也丢失，需要重新授权。

## JobScheduler

在API21及以上可以用jobscheduler，包括一次性或周期性两种job，可惜在API24以上，周期性job的周期最少为15分钟，所以有些场景不能用，只能用一次性job+重新调度规避。不管是哪种job都会被doze优化，doze后不再调度，即使加省电优化白名单也不行。

一次性job，如果加省电白名单，可以一解锁就立刻触发，如果不加省电白名单，锁屏时间一长，解锁后就需要过段时间才能重新触发，这个时间很诡异，有长有短。但切换一下网络就很容易再次触发。一次性job有两个重要参数：

* latency是等待时间，可以用这个加上重新调度来模拟周期job
* deadline是指即使条件不满足时，隔多久也会强制触发，deadline甚至可以小于latency，或者0表示不会强制触发。但如果此时处于锁屏，仍然不行。

总之，JobScheduler不能在锁屏下触发，并且如果不加省电白名单，解屏后有一段诡异的不触发时间。 Job除了时间限制外，可以加一些限制条件，比如：
* Network type 
* Charging
* Idle       这个不是dumpsys的那个idle，这个idle是指屏幕关闭后的一段时间，荣耀8x下实测下来很难触发 

如果有多个条件，是与关系。


## 设备状态

可以用下面的命令查询当前设备状态：

* adb shell dumpsys deviceidle | grep mState

也可以用adb shell dumpsys deviceidle step，手工向后步进状态，直到IDLE和IDLE_MAINTENANCE。

## 设备id
* Android ID，不需要权限，但刷机和设备重置后会变。
* IMEI，Android6之后需要READ_PHONE_STATE 权限，10.0之后有权限也读不了了。
* Build.SERIAL，直接访问在8.0之后总是返回“unknown”，需要READ_PHONE_STATE权限，然后调用Build.getSerial()，但10.0之后也不行了。
* wifi的mac地址，目前10.0还可用,如果重启手机后，Wifi没有打开过，是无法获取其Mac地址的?
```
public static String getWifiMac() {
    try {
        Enumeration<NetworkInterface> enumeration = NetworkInterface.getNetworkInterfaces();
        if (enumeration == null) {
            return "";
        }
        while (enumeration.hasMoreElements()) {
            NetworkInterface netInterface = enumeration.nextElement();
            if (netInterface.getName().equals("wlan0")) {
                return formatMac(netInterface.getHardwareAddress());
            }
        }
    } catch (Exception e) {
        Log.e("tag", e.getMessage(), e);
    }
    return "";
}
```