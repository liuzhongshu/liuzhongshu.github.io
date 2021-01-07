---
published: true
title: Java性能测试
layout: post
---


主要是CPU、Memory、Storage几个方面

## CPU
* 不能用volatile，如果用volatile，大部分语句中都会涉及大量的memory操作，cpu测试就变成memory测试

## Memory

* 用System.arraycopy, 比for循环快的多，可以准确测试出内存带宽。
* 单线程即可，因为相对memory，CPU足够快了。

## Storage

* 和Memory一样，单线程即可。
* 写速度比较好测试，但是os有delay write，所以如果想真实速度，可以用java的stream.getChannel().force(false); 这样测下来flash速度不太稳定，估计和flash有时需要擦除有关。
* 读速度受缓存影响极大，现代os的缓存非常智能，所以一定要清缓存，否则测试的就是memory速度，但是android下缓存怎么清？
