---
published: true
title: Python XML
layout: post
---

Python的html/xml解析工具有很多

* ElementTree
* BeautifulSoup
* lxml

### ElementTree

ElementTree是Python内置的XML解析工具因此无需第三方库

* 简单的xpath支持

### BeautifulSoup

BeautifulSoup对html有很好的兼容性，不要求严格的xml，这对于网页解析比较方便，支持css选择器，但不支持XPath

### lxml

使用C扩展的xml解析库，性能最佳，不过要小心用在html解析上，因为很多html，充斥着不规范的xml格式，导致解析异常。

比较好的Python爬虫使用

* request 替代urllib和urllib2
* BeautifulSoup ?

