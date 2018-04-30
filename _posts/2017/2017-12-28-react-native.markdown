---
published: true
title: React Native
layout: post
---

React Native是跨平台开发的一个常见选择，其他选择有Codova等，本文介绍React Native（下面有时缩写为RN），RN的Getting Started通常有两个选择：CRNA或常规工程，下面一一介绍：

## CRNA工程

```
npm install -g create-react-native-app
create-react-native-app rnTest
```

这种方法创建的工程，称之为CRNA工程，非常轻量，不需要预安装Android或iOS开发环境，所以非常适合初学，但是因为没有开发环境，所以手机上需要预安装一个工具Expo（android或iOS都可以），用来从开发机上同步代码，如果已经安装Expo，那么就可以做接下来的步骤

```
cd rnTest
npm start
```

会在终端上出现一个大大的二维码，然后用Expo去扫这个二维码即可加载了（确保手机和开发机在同一局域网下）,我发现这个二维码经常扫不出，可以手工输入地址来解决。如果不愿意下载安装Expo，可以用USB连接好手机，用npm run android，RN会尝试用adb工具下载Expo到手机中，这样可以省去下载Expo的过程，但是需要开发机上有android开发工具（不需要编译，但需要adb工具）。

总之，运行起来项目之后，就可以尝试有趣的hot reload了，按手机的菜单键（没有这个键，可以摇动手机），会出现开发者菜单，里面有live reload和hot reload，前者相当于全量刷新，后者相当于保持状态下增量注入，缺省情况下hot reload是禁止的，可以打开，这样代码更新后刷新的速度比live reload要快一些，和Cordova下的hot reload一样，有时hot reload工作会不正常，需要手工reload。

当然，一旦expo安装好，以后是不需要usb连接了，比较方便。但这种CRNA工程有个缺点就是不能包括Native代码，只能使用RN自带的API和组件，因为这些组件已经包括在Expo的客户端里，所以如果你确信你的APP需要Native模块，可以直接走下面的常规工程。

## 常规工程

也称之为Native工程，这种工程需要本机有android或iOS开发环境，并且在开发过程中一般使用有线（USB）连接真机。

```
react-native init xxx
cd xxx
react-native run-android
```

注意：运行的命令和CRNA工程不一样，不再是npm run android。把CRNA工程转换为常规工程的方法是：
```
npm run eject
```

## 调试
无论CRNA还是普通工程，调试都是通过chrome，在开发者菜单中选择debug js remotely，就会在桌面上自动启动chrome的调试页了。开启chrome的开发者工具页面后，按下图的路径找到被调试的js就可以了。

![](../../public/images/2018-03-04-21-53-26.png)

在调试版本，经常会看到黄色的警告提示，如果不想看到这些警告，在js代码入口处增加：

```
console.disableYellowBox = true;
```

## 发布

一个最简单的RN工程，编译出的apk也大概有8M，如果想缩小体积，可以考虑[删除x86的支持](https://stackoverflow.com/questions/35940957/how-to-decrease-react-native-android-app-size)，大概可以缩小到4M多。

## 控件
控件的描述使用了一个Javascript扩展JSX，因此写起来才比较方便，避免了在JS里“嵌入”用字符串括起来的大段模板，这点和Vue不一样，Vue是把模板单独作为一段，和JS平行起来构成了VUE文件。

控件的安装流程为：

* constructor(object props) 
* componentWillMount() 
* render() 
* componentDidMount()

控件的刷新流程为：

* componentWillReceiveProps(object nextProps) 
* shouldComponentUpdate(object nextProps, object nextState) 
* componentWillUpdate(object nextProps, object nextState) 
* render() 
* componentDidUpdate(object prevProps, object prevState)

### 属性Props
```
class Greeting extends Component {
  render() {
    return (
      <Text>Hello {this.props.name}!</Text>
    );
  }
}

export default class LotsOfGreetings extends Component {
  render() {
    return (
      <View style={{alignItems: 'center'}}>
        <Greeting name='Rexxar' />
        <Greeting name='Jaina' />
      </View>
    );
  }
}
```

A parent element may alter a child element's props at any time. The child element will generally re-render itself to reflect its new configuration parameters. A child component may decide not to re-render itself even though its configuration has changed, as determined by shouldComponentUpdate() (more on this in the Component Lifecycle API section).

### 状态 State

属性是不变的，状态是可变的，文档中的这个例子很清楚：
```
class Blink extends Component {
  constructor(props) {
    super(props);
    this.state = {isShowingText: true};

    // Toggle the state every second
    setInterval(() => {
      this.setState(previousState => {
        return { isShowingText: !previousState.isShowingText };
      });
    }, 1000);
  }

  render() {
    let display = this.state.isShowingText ? this.props.text : ' ';
    return (
      <Text>{display}</Text>
    );
  }
}
```
注意: 状态在构造函数中可以直接赋值，在method中可以读取this.state, 但只能通过setState来修改，每次修改会自动触发控件的刷新。如果不需要自动触发刷新，可以不通过state来管理状态，可以直接在this里创建新的变量。

### 样式
类似CSS，但去除了CSS中‘难用’的Cascade带来的一些问题，RN中样式的一个常见用法是把样式定义为一个数组，在代码中引用不同的样式名就可以了，达到了Web上class的效果，比如：
```
var Style = StyleSheet.create({
  rootContainer: {
    flex: 1
  },

  displayContainer: {
    flex: 2,
    backgroundColor: '#193441'
  },

  inputContainer: {
    flex: 8,
    backgroundColor: '#3E606F'
  }
}
...
render() {
    return (
      <View style={Style.rootContainer}>
        <View style={Style.displayContainer}></View>
        <View style={Style.inputContainer}>
          {this._renderInputButtons()}
        </View>
      </View>
    )
  }

```
有时，针对不同平台需要不同的样式，可以通过Platform来选择，举例：

```
import { Platform, StyleSheet } from 'react-native'
const styles = StyleSheet.create({
  container: {
    fontFamily: 'Arial',
    ...Platform.select({
      ios: {
        color: '#333',
      },
      android: {
        color: '#ccc',
      },
    }),
  },
});
```

### 布局
主要的布局方式兼容Web的Flex布局，但比Web要简单并且好用,参考下面的文档：

* https://medium.freecodecamp.com/an-animated-guide-to-flexbox-d280cf6afc35

简单的说，Flex容器的几个属性：
* display: flex; 申明容器使用flex布局，这是唯一一个必选设置
* flex-direction 控制主轴方向，row为横排（缺省），column为竖排
* flex-wrap 控制主轴方向溢出如何处理，缺省为滚动，wrap则折行
* justify-content 控制主轴的布局，可选flex-start（缺省），flex-end，center，space-between，space-around
* align-items 控制主轴垂直方面行内的布局，可选flex-start，flex-end，center，stretch，baseline （因为在垂直方向只有一个元素，所以没有space控制）
* align-content 控制主轴垂直方面多行的布局，当然如果没有多行就不起作用了。

容器只对其下层的第一级元素进行布局，Flex元算的属性：
* flex-grow 容器大于元素时，额外空间占比
* flex-shink 容器大于元素时，收缩空间占比
* flex-basis 初始元素大小，可以为auto
* flex 由上面三个属性组合出来，如果没有flex，缺省为0 1 auto，如果有flex简写，比如flex 2，则等价于flex 2 1 0%，这点比较奇怪，但是最好计算，因为basis为0，则无需考虑shink。
* align-self 对容器的align-items在子元素上的重载


除了Flex，另一个常用的布局方法是使用Dimensions，参考下面：

```
import { Dimensions } from 'react-native'
const { width, height } = Dimensions.get('window')
```


#### 隐藏状态栏
很简单，只需要在根view中加入：

```
<StatusBar hidden />
```

#### 应用名称、图标和版本
相对于Cordova，RN对应用名称和图标几乎没有封装，修改起来非常不便，好在都有工具可以做到，下面是做法：

* 修改图标
```
npm install -g yo generator-rn-toolbox
yo rn-toolbox:assets --icon .\src\assets\icon.png
```
注意修改图标需要预先安装image-magick，修改名称就比较麻烦了，简单的方法是删除android和ios目录后，重新react-native eject

* 修改package名称

```
npm install react-native-rename -g
react-native-rename "myapp" -b com.mycompany.myapp
```
*修改版本号

只能手工修改，或者通过build.gradle文件自动化，参考[这里](https://stackoverflow.com/questions/35924721/how-to-update-version-number-of-react-native-app)

### 路由和导航
和VUE一样，路由模块不属于核心模块，不过大部分RN项目使用了某个路由模块，其中最流行的为[React Navigation](https://reactnavigation.org)。它提供了最基础的StackNavigator模块，这个类似Web导航（Stack的概念），差别就是StackNavigator还提供了切换时的动画。

StackNavigator本身是一个函数，提供两个参数，一个配置参数配置路由表，一个选项参数，举例：
```
const RootStack = StackNavigator(
  {
    Home: {
      screen: HomeScreen,
    },
    Details: {
      screen: DetailsScreen,
    },
  },
  {
    initialRouteName: 'Home',
  }
);
```
这样返回的组件就可以作为根组件了，同时会在每个screen组件下注入一个属性navigation用于跳转，比如`this.props.navigation.navigate('Details')`或者`this.props.navigation.goBack()`, 跳转的时候当然是可以带参数的，可以放在后面，比如：

```
 this.props.navigation.navigate('Details', {
    itemId: 86,
    otherParam: 'anything you want here',
  });
```            
在DetailsScreen组件中，可以通过`this.props.navigation.state`来读取传入的参数。

### TabNavigator
除了StackNavigator，另一个常用的为TabNavigator，就是很多应用中首页中看到的多Tab的设计。

## 自带组件

### 按钮
比如跳转打开浏览器的按钮可以这样定义：
```
<TouchableHighlight
  onPress={() => Linking.openURL(
    `http://finance.yahoo.com/q?s=${this.state.symbol}`
  )}>
```

### 列表
RN内置两个列表控件，ListView和FlatList，建议使用新的FlatList，性能更好，并且将来应该会替代ListView

## 第三方组件
除了RN自带的组件，还有一些重要的第三方组件：

### react-native-vector-icons
* yarn add react-native-vector-icons
* react-native link
* 在android/app/build.gradle的dependencies下加入compile project(':react-native-vector-icons')

还可以找到很多第三方组件库（类似Cordova下可以用Quasar等库），比如：
* [react-native-elements]()
* [nativeBase](https://nativebase.io/)
* [react-native-scrollable-tab-view](https://github.com/skv-headless/react-native-scrollable-tab-view)
* [rn-viewpager](https://github.com/zbtang/React-Native-ViewPager)

## 其他有趣的
* [storybook](https://github.com/storybooks/storybook) RN快速原型工具
* [react-canvas](https://github.com/Flipboard/react-canvas) 一组高性能组件
* [ignite](https://github.com/infinitered/ignite) 快速RN模板项目命令行工具
* [30-days-of-react-native](https://github.com/fangwei716/30-days-of-react-native) 30天RN教程

### 坑

总的来说RN的体系坑比较多，比Cordova环境难一些，从github上找一些RN的项目，大多在编译时都会遇到各种各样的问题，下面是一些我踩过的：

### 编译时 Could not delete path 'android\app\build\generated\source\r\release\android\support'
不知道原因是什么，但是有时会出现这个问题，通常重新编译就好了，实在不行，需要cd android && gradlew clean，再重新来过

### 编译时com.android.ddmlib.InstallException: Failed to install all
我在一台android手机上遇到了这个问题，尝试下面几个方法：
* react-native run-android --deviceId xxx //后面的xxx是adb devices命令返回的设备id
* 如果是小米，关闭MIUI优化
* 还不行，换一台手机

### 运行Android debug版本不覆盖手机老版本
这个还没有特别好的解决方法，只能先在手机上卸载，或者自己在构建脚本中加入adb uninstall命令。

### 运行时报错 set canOverrideExistingModule=true
在MainApplication.java(android/app/src/main/java/../..)下找看看，去除getPackages函数下重复的MapsPackage和import，这应该是RN的bug，也许未来的版本会解决。

### 运行Android debug版本一段时间后出现could not connect to development server
packager的端口断了，解决方法是再次运行adb reverse tcp:8081 tcp:8081

### Debug版本打不开开发者菜单
有些手机没有菜单键，可以通过下面几个方法尝试：

* 摇动手机
* adb shell input keyevent 82
* 在手机上安装可以模拟菜单键的悬浮球软件

## 参考
* https://hackernoon.com/learning-react-native-where-to-start-49df64cf14a2
* http://www.reactnativeexpress.com
