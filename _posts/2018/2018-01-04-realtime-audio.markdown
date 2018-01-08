---
published: true
title: Realtime audio
layout: post
---

尝试做一个跨平台的实时音频处理应用(计算音高），发现一些坑，记录一下。

## Cordova 和 WebAudio
因为在浏览器其中已经看到基于WebAudio的成熟[应用](https://github.com/cwilso/pitchdetect)，所以首先想到的自然是Cordova和WebAudio的组合，WebAudio本身规格强大，如果完全支持的话，几乎不再需要额外的插件了。不过简单的尝试就发现WebAudio在Cordova下，Android和iOS都支持的不好，Android至少需要5.0，iOS即使最新也不支持。有文章说可以使用集成CrossWalk替换缺省的WebView，可以大幅提高WebAudio的兼容性，但以下两个原因，让我放弃了：

* 即便用CrossWalk，iOS下也不能完整支持WebAudio
* 集成CrossWalk会使得安装包增加30M

## Cordova插件 加 部分WebAudio
如果把WebView不支持（或兼容性不好）的部分用Cordova插件替代，主要是替代getUserMedia部分，即获取原始数据的部分。再结合WebAudio中相对简单的算法部分（兼容性较好），插件使用[cordova-plugin-audioinput](https://github.com/edimuj/cordova-plugin-audioinput)。

不过实际测试告诉我，WebAudio在移动端，尤其是iOS支持太差了，即便是算法部分，也在iOS下支持的很差，Android倒是基本没有问题了。可能是苹果不希望开发者使用WebView的方案吧，这条路也是很难行通。

## Cordova插件加纯JS算法
使用这个方案，就可以彻底抛弃了WebAudio，使用下面这个算法库：
* [pitchfinder](https://github.com/peterkhayes/pitchfinder)

## 性能
通常的音频采样率为44100，这个采样率下，如果做实时音频处理，对CPU的要求还是较高的，我测了在苹果的A5X CPU下，就很卡了，当然JS本身性能不佳也是一个原因，所以如果性能不够用，降低采样率是最简单的方法，比如下面的代码，将audioinput的采样率降到了8000，每100ms回调一次：

```
audioinput.start({
    sampleRate: 8000,
    bufferSize: 800 
});
```

如果你的算法不允许降低采样率，那似乎只能考虑将算法移植到原生代码里了。