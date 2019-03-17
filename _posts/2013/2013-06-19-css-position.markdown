---
published: true
title: CSS position和display
layout: post
---
CSS规范中position的定义可以取四种值，并且又和其他属性，比如display，LRTB(left,right,top,bottom的缩写)相关，所以不容易理解清楚，下面是我的解释：

# position

* static - 默认值，对象遵循HTML定位规则,此时LRTB无效。 

* relative - 对象将依据LRTB等属性在正常位置基础上偏移。

* absolute - 将对象从文档流中拖出，使用LRTB等属性相对于其最接近的一个有定位设置的父对象进行绝对定位。如果不存在这样的父对象，则依据 body 对象。absolute定位时，margin失效。

* fixed - 类似absolute，但是却直接相对于浏览器窗口，与父无关。

如果单从定义上看，这四种区别还比较模糊，尤其是relative和absolute是有点晕的，但结合应用场景，就很明确了，通常大部分元素都是static，如果想将某个元素脱离出来绝对定位，可以用absolute或者fixed，这时要看你希望相对的点，如果希望相对浏览器窗口固定，用fixed，如果希望相对父元素固定，用absolute。

但是注意absolute所相对的这个父元素必须不是static，通常就指定一下relative就好了，因为relative本身不改变元素的位置（前提是没有通过LRTB偏移），CSS这样设计的用意很容易理解，可以灵活的控制absolute元素想相对的元素，比如，想相对父的父，只要父为static，而父的父指定一下relative就可以了。

LRTB四个元素是可以同时用的，同时用的时候，如果元素大小未定，则修改元素大小，如果元素大小已经确定，则bottom和right被忽略。

# display

* display: none 不显示，也不分配空间（相当于元素从页面上删除）
* display: inline 一般的，span缺省使用这个display，这种元素不能设置height，width，padding、margin也不会影响他的占位（但会影响背景色）
* display: inline-block，可以设置box的hwpm了
* display: block，相对inline-block，尾部会自动换到下一行，div的缺省是block
* display: flex Flex布局，参考下面

# flex

简单的说，Flex容器的几个属性：
* display: flex; 申明容器使用flex布局，这是唯一一个必选设置
* flex-direction 控制主轴方向，row为横排（缺省），column为竖排
* flex-wrap 控制主轴方向溢出如何处理，缺省为滚动，wrap则折行
* justify-content 控制主轴的布局，可选flex-start（缺省），flex-end，center，space-between，space-around
* align-items 控制主轴垂直方面行内的布局，可选flex-start，flex-end，center，stretch，baseline （因为在垂直方向只有一个元素，所以没有space控制）
* align-content 控制主轴垂直方面多行的布局，当然如果没有多行就不起作用了。

容器只对其下层的第一级元素进行布局，Flex元算的属性：
* flex-grow 容器大于元素时，额外空间占比
* flex-shink 容器小于元素时，收缩空间占比
* flex-basis 初始元素大小，可以为auto
* flex 由上面三个属性组合出来，如果没有flex，缺省为0 1 auto，如果有flex简写，比如flex 2，则等价于flex 2 1 0%，这点比较奇怪，但是最好计算，因为basis为0，则无需考虑shink。
* align-self 对容器的align-items在子元素上的重载

# 居中

经常的一个需求是让div里的button居中，严谨的说flex布局可以做到，但更简单的做法就是一个text-align:center; 这个很方便。

如果是fixed元素想居中，方法如下：

```
position: fixed;top: 50%;left: 50%;transform: translate(-50%,-50%);
```