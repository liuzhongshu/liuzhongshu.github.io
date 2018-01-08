---
published: true
title: game engines
layout: post
---

## Unity

最广泛使用的一个引擎了（另一个是Unreal4），实际上Unity没有真正的2D引擎，他的2D部分是通过3D引擎模拟的，这点需要注意，如果是3D，Unity是非常安全的选择，如果是2D可以往下看。

## Godot
全功能2D/3D引擎(相对Unity还很轻量），内置GDScript（类python语法），Godot引入一些概念：
* 每个游戏有多个Scene，在Godot IDE内每个Scene可以独立运行，方便调试
* Scene由一组node或，形成一个tree，也可以把其他scene instance出来作为子节点。
* 每个node可以包含或引用到一个script
* UI组件很丰富，实际上Godot编辑器就是用自己的UI组件实现的

目前来说Godot有一个3.0Beta版本，和2.0相比，有很多[不兼容的地方](https://docs.google.com/spreadsheets/d/1SqLGKpF5B5KzYnY2JzuuP71tsR8WeXZn1imMvHkoKDc/edit#gid=0)（工程文件都不兼容），所以新开发者可以直接考虑从3.0入手，差别有：
* GDScript有了很明显的简化，比如```get_node("name").set_pos(get_node("name").get_pos() + Vector2(1,0))```可以简化为```$name.position.x += 1```
* 3.0增加了更多语言，可以使用C#等语言来开发了，这里甚至已经有了[性能对比](https://github.com/cart/godot3-bunnymark)。
* 物理引擎增加了Bullet
* 引擎的大小也从压缩后6M到9M了，根据需要选择。

### Godot的一些组件选择：
Godot的组件很多，大体分为三类，2D，3D和control，可以很容易从图标上区分，它们分别为蓝色、红色、绿色。2D游戏一般来说：

* 主角可以用Area2D
* 机器人可以用RigidBody2D
* 菜单等UI可以用CanvasLayer，下面添加各种Control


## Love
2D游戏引擎，非常轻量，Love不提供任何编辑工具，你可以在画布上做任何代码的控制，有些人很喜欢这种工作方式。

## 总结

最后是上述引擎的对比：


|      | 3D    | 语言   | license |平台 | zip大小| UI组件 |
|----- |-------------   | ------- |  --| -- | --|--|
| Unity|2D/3D  | C#     |  limited|all
| Godot|2D/3D  | GS/C++ |   FREE  |all| 6m/9m |内置
| Love | 2D    | Lua/C  |   FREE  |all|3m|
