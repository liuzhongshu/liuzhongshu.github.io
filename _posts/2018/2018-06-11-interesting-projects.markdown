---
published: true
title: 一些有趣的项目
layout: post
---

## flutter
google的下一代跨平台开发框架

## [annyang](https://github.com/TalAter/annyang)
封装浏览器的语音识别api，已经在桌面的chrome下支持比较完善了。

* 支持中文
* 都是云端做的，不支持离线识别，国内需要proxy，速度也比较慢
* 如果tab页面放到后台，可以识别
* 如果两个tab都开启识别，当前tab页会获得输入

示例代码：
```
  var commands = {
    '你好': function() { alert('Hello!'); }
  };

  annyang.setLanguage('zh-CN');
  annyang.addCommands(commands);
  annyang.start();
 ``` 

## [mmkv](https://github.com/tencent/mmkv)

腾讯开源的替代SharedPreferences的东东，居然支持iOS和win32.

## 其他
* 图像相似度算法 https://github.com/Nakilon/dhash-vips
* 古诗词数据库 https://github.com/chinese-poetry/chinese-poetry
* unicode大全 https://unicode-table.com
* [网易修改的duilib](https://github.com/netease-im/NIM_Duilib_Framework)
