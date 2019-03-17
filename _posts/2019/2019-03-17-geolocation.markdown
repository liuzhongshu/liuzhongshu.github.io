---
published: true
title: 手机获取地理位置
layout: post
---

开发cordova应用，如果需要获取位置，大部分人经过搜索会直接使用[cordova-plugin-geolocation](https://cordova.apache.org/docs/en/latest/reference/cordova-plugin-geolocation/), 可在国内，这个实际上是个坑，因为这个插件在安卓上直接使用了webview的位置API，而w3c的这个API在手机端，非常的不稳定，经常更新不了位置（返回之前的位置），起码我测试的两款手机（华为和魅族）都有这个问题，有时候重启手机可以解决问题，但显然不是个好方案。

## 替代方案

多次尝试之后发现一个替代方案，cordova-plugin-advanced-geolocation，在安卓上走原生接口，没有用w3c实现。

## 坐标系

获得的GPS坐标，通常不能直接用于地图，因为有三种坐标系统，需要做转换

* wgs坐标，用于硬件、谷歌地球 
* gcj坐标，用于高德地图、腾讯地图、谷歌地图的国内部分
* 百度坐标，又称bd09，用于百度地图

上面提到的插件获得的是wgs坐标，所以当用于地图的时候要根据你要用的API或场景，进行转换，不管是Java还是Javascript实现这个转换并不难，写几个函数就可以了。
