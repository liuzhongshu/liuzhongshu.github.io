---
published: true
title: Typescript
layout: post
---


ts是未来JavaScript的方向, typescript是js的超集，所以任何js代码都是合法的ts代码，但是增加了类型检查，可能会导致不做修改的js代码报错，当然这取决于原先js代码的写法。

一旦ts编译完成，ts会生成纯粹的js代码，类型代码会被移除，js的运行行为不做任何改变，也不需要额外的runtime，所以浏览器和其他js引擎无需任何修改。


## 类型
typescript可以隐式推导类型，比如：

```
let helloWorld = "Hello World";
```

## union、enum、turple

```
type LockStates = "locked" | "unlocked"; //union
enum Color {
  Red = 1,
  Green = 2,
  Blue = 4,
}
let c: Color = Color.Green;  //enum
let x: [string, number]; //turple
```

## array和generics

```
let list: number[] = [1, 2, 3];
let list: Array<number> = [1, 2, 3];
type StringArray = Array<string>
```

##类型检查

可以用typeof在运行时刻查询类型，比如：
```
if (typeof var === "string") ...
```

## unknown和any
any比较好理解，就是和老的js相同，不做任何检查。unknown是typesafed any，也就是说unknown在使用的时候必须先检查好类型再使用。


## void、never
void一般用于函数，表示无返回值，never则是指函数抛异常或者死循环，never的作用主要是加强类型检查。