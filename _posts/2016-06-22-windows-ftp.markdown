---
published: true
title: Windows FTP编码
layout: post
---
Windows使用资源管理器访问ftp是个很方便的功能，遗憾的是资源管理器似乎不能正确发送utf-8文件名，我做了一个简单的测试，使用Apache FTP sever做服务器端，资源管理器发送“开发环境.docx”这个文件，服务器端会报错：

Client sent command that could not be decoded: 53 54 4F 52 20 E5 BC 80 E5 8F 91 E7 8E AF E5 A2 3F 64 6F 63 78 0D 0A

使用十六进制编辑器比对一下，客户端发来的bytes确实不正确（对了前面三个中文，第四个字错了）。让我感到非常吃惊的是，如果确实是个bug，这个bug竟然从windows7/windows8.1/一直延续到了Windows 10，在微软的论坛上，可以看到这个[bug](https://social.technet.microsoft.com/Forums/en-US/6b4df752-51d1-42fb-baf1-8600fa0bdfc5/utf8-encoding-bug-report-about-using-ftp-with-windows-explorer?forum=w7itpronetworking) 在2012年已经有人提出（也许还有更早的帖子），可是并没有官方回复。


## 规避方法
资源管理器在连接FTP服务器时，会通过opts utf8 on来协商编码，一般来说，支持UTF-8的服务器会回应这个协商，但是对于文件名错误的问题，也许强制服务器端编码为gb是一个规避方法。注意到一些Android端文件管理器也支持FTP Server功能，为了能支持windows资源管理器，魅族的文件管理器允许设置编码，并且缺省为GBK（规避了上面的这个问题）。

另一个方法当然就是不用资源管理器，安装第三方ftp客户端了。