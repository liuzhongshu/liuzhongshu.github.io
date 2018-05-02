---
published: true
title: 邮件工具箱
layout: post
---

## 邮件转微信

这个很简单，微信可以绑定qq邮箱，给qq邮箱发邮件就可以有微信提醒了。

## 给自己发邮件

简单的方法是使用IFTTT提供的webhook，只需要触发一个http请求就可以，但实测下来并不可靠，有时会丢。

可靠的方法还是通过SMTP，SMTP能发送的数量肯定是受限的，另一个缺点是，不是每个场合都能能直接使用SMTP，不是很方便，继续往下看。

## 群发

毕竟不是用于机器发送的。SMTP不能解决群发，发多了要被禁止（包括Gmail也是有每天发送数限制）。邮件群发服务商可以做群发，但大多要收费，不过有些也提供免费的限额，比较常用的有mailgun和sendgrid。

### mailgun集成namecheap
如果想通过自己域名来群发，需要在域名处修改DNS记录，可惜文档写的不明确，参考[这个](https://medium.com/@ustcboulder/setup-mailgun-on-namecheap-spf-dkim-cname-and-mx-684b5c1fb492)可以解决mailgun和namecheap集成的问题。