---
published: true
title: chrome app体验
layout: post
---

因为一个app需要用到CORS，想着chrome 扩展app应该可以绕过CORS，就尝试了一下，可惜结论不太好，我根本没办法移植我的vue应用到chrome app，原因记录下来：

## localStorage 
chrome app不支持localStorage，必须改写为chrome.storage，而且这个chrome.storage并不是改个名字这个简单，而是api都变了，从同步改成了异步，我自己的代码，花了一点时间改正过来，可是第三方库的代码，就无能为力了，不知道为什么chrome要引入这个不兼容。

## history.pushState
这个是vue-router是常用的API，同样也不兼容。

没有特别的动力，上面两个原因已经足够让我放弃移植的想法了。