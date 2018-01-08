---
published: true
title: React Native
layout: post
---

跨平台开发的一个常见选择是React Native，其他选择有Codova

## 安装

```
npm install -g create-react-native-app
create-react-native-app rnTest
```

这种方法创建的工程，称之为CRNA工程，非常轻量，不需要预安装Android或iOS开发环境，所以非常适合初学，但是因为没有开发环境，所以手机上需要预安装一个工具Expo（android或iOS都可以），用来从开发机上同步代码，如果已经安装Expo，那么就可以做接下来的步骤

```
cd rnTest
npm start
```

会在终端上出现一个大大的二维码，然后用Expo去扫这个二维码即可加载了（确保手机和开发机在同一局域网下）。如果不愿意下载安装Expo，可以用USB连接好手机，用npm run-android，RN会尝试用adb工具下载Expo到手机中，这样可以省去下载Expo的过程，但是需要开发机上有android开发工具（不需要编译，但需要adb工具）。

总之，运行起来项目之后，就可以尝试有趣的hot reload了，按手机的菜单键，会出现开发者菜单，里面有live reload和hot reload，前者相当于全量刷新，后者相当于增量注入，当然一旦expo安装好，以后是不需要usb连接了。

## 调试

