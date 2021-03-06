---
published: true
title: Octopress
layout: post
---
Update(2016): 现在我已经不推荐Octopress了，对大部分人，使用Jeklly和TinyPress的组合更方便一些。

---

[Octopress](http://octopress.org)是非常方便的静态博客程序，尤其适合github。

### 安装

```
git clone git://github.com/imathis/octopress.git octopress
cd octopress
bundle install
rake install
```

这就把Octopress装好了，上面的目录名octopress可以任意更改，接下来就需要设置github绑定了：

```
rake setup_github_pages
```

上面的命令会有提示输入git路径，注意要输入git的全路径，比如git@github.com:xxx/yyy.git

### 使用
新增博客帖子很简单，可以使用命令：

```
rake new_post["title"]
```

当然也可以在source/_posts目录下手动添加博客，要编译生成静态页面，并上传到github

```
rake generate
rake preview
rake deploy
```

如果你的ruby gem比较复杂，则需要在命令前使用bundle exec。

如果要上传源文件：

```
git add .
git commit -m "source commit"
git push origin source
```

### 美化
如果喜欢某个样式，可以使用git submodule来安装：

```
git submodule add git://github.com/octopress-themes/xxx.git .themes/xxx
rake install['xxx']
```

这样做的好处是，可以随时更新theme

```
cd .themes/xxx
git pull origin master
```

### 修改样式
octopress的theme一样体现了黑客风格，使用sass来做预编译。sass在这里物尽其用，使得修改theme更容易和一致。先看总体的文件：

```
@import "compass";
@include global-reset;
@include reset-html5;

@import "custom/colors";
@import "custom/fonts";
@import "custom/layout";
@import "base";
@import "partials";
@import "plugins/**/*";
@import "custom/styles";
```
可见几个目录的分工，custom用于在theme的基础上做用户定义，plugins是插件，基本的theme定义放在base和parials下。将custom分为两部分，一部分在base前，一部分放在最后是有讲究的，放在base之前的都是一些sass变量定义，因为sass在变量定义的时候支持前面的定义覆盖后面的定义（使用sass的!default来实现），所以这部分变量定义的部分要必须在前面。最后的custome/styles定义的一些样式，按照css的规格，后定义的样式覆盖前面的样式，所以这部分又必须放在后面了。

一般来说custom和plugins部分的内容都是空的，所以主要的theme定义就是在base和partials两个子目录下。base的加载顺序是这样的：
```
@import "base/utilities";
@import "base/solarized";
@import "base/theme";
@import "base/typography";
@import "base/layout";
```
显然，从命名上大致可以看出，utilities部分是一些公共函数。solarized定义一些颜色，theme是一些可调整变量，typography定义字体，最后的layout有点奇怪，他准确的说是整体布局，通过media query定义了一些全局并且可变的布局。具体的布局样式的定义其实都在另一个目录partials下，partials的加载就无所谓顺序了，针对各页面的组成做相应的细节定义，比如header，footer等。

在开始我自己的theme之前，我需要一些灵感，可以在[这里](http://opthemes.com/)获得。

### 缺点

Octopress需要整套ruby的环境。