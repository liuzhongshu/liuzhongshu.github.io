---
published: true
title: 文件压缩
layout: post
---

开发过程中经常需要对一些资源文件压缩，已减小文件体积，这里记录一二。

## mp3

用ffmpeg就可以对mp3进行压缩了，最常见的cbr，vbr模式，ffmpeg都支持，二者质量的参数不一样，对于cbr，直接使用 -b:a 指定比特率就可以了，可选8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320，后面价格k就可以了，对于vbr，一共有10个质量等级，从-q:a 0到-q:a 10, 下面是两个例子：

```
ffmpeg -i input.mp3 -q:a 10 output.mp3
ffmpeg -i input.mp3 -b:a 16k output.mp3
```

我的测试结果，对于人声文件，我能结受的最低质量就是 -q:a 10和-b:a 16k，-q:a 10产生的平均比特率大约20k，所以要想得到最小的文件，我通常使用-b:a 16k，虽然-b:a 8k可以再减小一半大小，但质量已经明显不可接受了。最后给出ffmpge的[官网说明](https://trac.ffmpeg.org/wiki/Encode/MP3)。

另一个常见的用来减小文件的手段是crop，就是掐头去尾，使用ffmpeg也很简单，下面是从1分10秒点2截到1分30秒的例子：

```
ffmpeg -i input.mp3 -ss 1:10.2 -to 1:30 -b:a 16k output.mp3
```

如果是截取一定长度，用-t替代-to就可以了。