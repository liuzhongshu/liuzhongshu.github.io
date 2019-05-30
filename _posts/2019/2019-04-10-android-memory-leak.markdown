---
published: true
title: Android内存泄漏
layout: post
---

在build.gradle中增加:
```
compile 'com.squareup.leakcanary:leakcanary-android:1.6.3'
```

Application类增加：
```
public class XXXApplication extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        if (LeakCanary.isInAnalyzerProcess(this)) {
            return;
        }
        LeakCanary.install(this);
    }
}
```

如果原来没有Application类，需要增加一个，并在AndroidManifest.xml的Application中注册：

```
android:name=".XXXApplication"
```

最后，使用debug run，但一定不能连接调试器，就可以动态监控内存泄漏了，但不是实时，要每过一段时间leakcanary会自动检测一遍。

