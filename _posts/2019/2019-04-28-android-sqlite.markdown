---
published: true
title: Android Sqlite
layout: post
---

## SQLiteOpenHelper

很多教程都是从SQLiteOpenHelper开始的，其实SQLiteOpenHelper是可选的，大概的过程为：

* 如果数据库已存在，打开
* 如果数据库不存在，创建
* 按版本，更新schema
* 设置版本

应用如果不涉及schema升级，可以不用SQLiteOpenHelper，直接用SQLiteDatabase.openOrCreateDatabase就可以了，这个才是底层的db接口。

## 打开外部数据库

缺省方式，数据库位于App内部存储，如果想打开存储卡上的数据，需要些一个特殊的context骗过openOrCreateDatabase接口。