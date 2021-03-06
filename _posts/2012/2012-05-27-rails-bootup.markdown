---
published: true
title: Rails快速指南
layout: post
---
### RVM

ruby的包管理器叫做gem，有些类似Linux的apt-get，leopard下已经预装gem，不过版本号实在不敢恭维:1.0.1，一个让人不太信任的数字，所以第一步就是刷新gem，不过对于系统已经安装的ruby和gem，替换不太好，幸好有rvm这个极好的工具，可以让你在个人的目录下管理ruby和gems。安装的方法就是下面的一条bash命令。

```
bash < <(curl -s https://raw.github.com/wayneeseguin/rvm/master/binscripts/rvm-installer)
```

上面是安装了rvm到你的个人目录下，要使用rvm，还需要设置bash的初始化文件，同样的，一条命令：

```
echo '[[ -s "$HOME/.rvm/scripts/rvm" ]] && . "$HOME/.rvm/scripts/rvm" # Load RVM function' >> ~/.bash_profile
```

别忘了重新登录，以生效上面的profile。

### Ruby和Rails

有了rvm，可以很方便的安装ruby了，如下：

```
rvm install 1.9.3
socksify_ruby localhost 8888 `which gem` install rails
```

理论上直接 gem install rails就可以了，不过因为一个你能懂的原因，我发现先装好socksify_ruby这个gem要好很多.

大部分书上介绍rails的第一条命令是：
```
rails new project
```

### 集成Capistrano
Capistrano是很好的部署方法，如果需要自己部署的话，建议使用Capistrano，首先要生成Capistrano的配置文件：

```
capify .
```
生产的deploy.rb，我们要做一些配置.

### application.rb
这个文件中，有一些全局的应用程序设置，很多情况下需要修改的是时区，默认为UTC

```
config.time_zone = 'Beijing'
```

### database.yml
配置数据库，大部分情况下，我们喜欢postgresql，所以：

```
production:
  adapter: postgresql
  encoding: unicode
  database: name
  pool: 5
  username: user
  password: pass
```

上面是设置的生产环境，开发环境我们有时会直接使用sqlite，sqlite和postgresql的兼容性很好，当然有时也会直接使用postgresql，那么就同时需要修改配置文件中的development段落的相关设置。

### 部署

需求
- VPS安装Ubuntu 11.04
- RVM安装ruby和rails
- Nginx和Passenger模块
- PostgreSQL

ruby要用1.9.3以上版本，因为1.9.3版本解决了一个性能问题，Rails用3.1以上版本，因为在之前的版本会有assets方面的兼容性问题。


安装 Nginx

```
gem install passenger
sudo apt-get install libcurl4-openssl-dev
rvmsudo passenger-install-nginx-module
```

安装nginx启动脚本

```
wget http://www.smashingmagazine.com/files/nginx
sudo mv nginx /etc/init.d/nginx
sudo chmod +x /etc/init.d/nginx
sudo /usr/sbin/update-rc.d -f nginx defaults
```

安装 PostgreSQL，偏爱PostgreSQL，除了PostgreSQL成熟以外，还因为PostgreSQL和Sqlite比较接近的SQL语句，这样可以在两者间随意切换。

```
sudo apt-get install postgresql libpq-dev
```