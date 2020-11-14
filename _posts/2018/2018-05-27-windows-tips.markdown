---
published: true
title: Windows问题汇总
layout: post
---

日常的开发都是在windows下，记录一些小问题的解决方法，目前用到系统是windows8.1，不过大部分对windows10应该也有效。

## update和defender
这两个是windows占用cpu最主要的可能原因，好在可以通过gpedit.msc禁止他们，所以不要安卓家庭版哦 ;)
 
* Computer Configuration > Administrative Templates > Windows Components > Windows update > Configure Automatic Updates
* Computer Configuration > Administrative Templates > Windows Components > Windows Defender Antivirus > Turn off Windows Defender Antivirus


## 多剪贴板

最新的windows 10居然支持了多剪贴板功能，在控制面版里使能以后，按Windows+V就可以触发了。

## 屏幕截图

windows 10内置了截图工具，可以通过右下角的通知栏来启动，也可以通过热键Win+Shift+S，如果这还不够用，可以用第三方工具，我喜欢picpick。

## chrome鼠标失灵
常发生在休眠重新唤醒之后，现象为左键点击标签页变成了关闭标签页，百搜不得其解，每次要重启才能恢复，后来偶然发现，按住鼠标中键几秒后松开就解决了。

## 输入法
我不喜欢win8的微软拼音，非常卡。更偏好手心输入法，但是安装完后，多余的微软拼音却无法移除，这点微软真的是太过小气，好在有下面的回旋之道：
* 在 控制面板》区域里 先把英语加回去
* 把英语放在第一位，这样可以把第二位的中文移除（也移除了所有的中文输入法）
* 再安装输入法，在安装完后，中文语音会被自动加回去，同时微软拼音终于消失了。

![](../../public/images/2018-06-09-09-47-10.png)

* 我一般会设置英文语言为默认，这样重新启动windows后，所有的应用都默认英文输入。
* 然后手心输入法里默认中文状态，这样切换到中文输入法后直接输入中文，并隐藏输入法的状态栏。
* 中英文切换，我不使用shift热键切换中英文状态，一个原因是shift太容易冲突，另一个是原因是输入法的中英文状态是个“局部”状态，切换到其他应用就丢失了，会出现中英文乱跳的情况，所以我干脆直接切换掉输入法，直接在英文键盘和中文输入法间切换，这个状态是可以全局保留的。

做了以上步骤后，输入法基本满足我的要求，但对于经常中英文混合输入的我来说，有个状态指示会更好一点（输入法状态条通常太远，且干扰界面显示，我不太喜欢。找了良久，决定用大小写指示灯试试，我找了一个autohotkey的脚本来说，重载Capslock键，切换输入法并点亮指示灯，代码如下：

```
; 使用capsLock切换输入法，CMode记录当前状态，并用LED指示
*CapsLock::
  send {LCtrl down}{LShift down}{LShift up}{LCtrl up}
  CMode:=!CMode 
  If CMode = 1
  {
    KeyboardLED(4,"on")
  }
  else 
  {
    KeyboardLED(4,"off")
  }
Return


/*
    Keyboard LED control for AutoHotkey_L
        http://www.autohotkey.com/forum/viewtopic.php?p=468000#468000
    KeyboardLED(LEDvalue, "Cmd", Kbd)
        LEDvalue  - ScrollLock=1, NumLock=2, CapsLock=4
        Cmd       - on/off/switch
        Kbd       - index of keyboard (probably 0 or 2)
*/

KeyboardLED(LEDvalue, Cmd, Kbd=0)
{
  SetUnicodeStr(fn,"\Device\KeyBoardClass" Kbd)
  h_device:=NtCreateFile(fn,0+0x00000100+0x00000080+0x00100000,1,1,0x00000040+0x00000020,0)
  
  If Cmd= switch  ;switches every LED according to LEDvalue
   KeyLED:= LEDvalue
  If Cmd= on  ;forces all choosen LED's to ON (LEDvalue= 0 ->LED's according to keystate)
   KeyLED:= LEDvalue | (GetKeyState("ScrollLock", "T") + 2*GetKeyState("NumLock", "T") + 4*GetKeyState("CapsLock", "T"))
  If Cmd= off  ;forces all choosen LED's to OFF (LEDvalue= 0 ->LED's according to keystate)
    {
    LEDvalue:= LEDvalue ^ 7
    KeyLED:= LEDvalue & (GetKeyState("ScrollLock", "T") + 2*GetKeyState("NumLock", "T") + 4*GetKeyState("CapsLock", "T"))
    }
  
  success := DllCall( "DeviceIoControl"
              ,  "ptr", h_device
              , "uint", CTL_CODE( 0x0000000b     ; FILE_DEVICE_KEYBOARD
                        , 2
                        , 0             ; METHOD_BUFFERED
                        , 0  )          ; FILE_ANY_ACCESS
              , "int*", KeyLED << 16
              , "uint", 4
              ,  "ptr", 0
              , "uint", 0
              ,  "ptr*", output_actual
              ,  "ptr", 0 )
  
  NtCloseFile(h_device)
  return success
}

CTL_CODE( p_device_type, p_function, p_method, p_access )
{
  Return, ( p_device_type << 16 ) | ( p_access << 14 ) | ( p_function << 2 ) | p_method
}


NtCreateFile(ByRef wfilename,desiredaccess,sharemode,createdist,flags,fattribs)
{
  VarSetCapacity(objattrib,6*A_PtrSize,0)
  VarSetCapacity(io,2*A_PtrSize,0)
  VarSetCapacity(pus,2*A_PtrSize)
  DllCall("ntdll\RtlInitUnicodeString","ptr",&pus,"ptr",&wfilename)
  NumPut(6*A_PtrSize,objattrib,0)
  NumPut(&pus,objattrib,2*A_PtrSize)
  status:=DllCall("ntdll\ZwCreateFile","ptr*",fh,"UInt",desiredaccess,"ptr",&objattrib
                  ,"ptr",&io,"ptr",0,"UInt",fattribs,"UInt",sharemode,"UInt",createdist
                  ,"UInt",flags,"ptr",0,"UInt",0, "UInt")
  return % fh
}

NtCloseFile(handle)
{
  return DllCall("ntdll\ZwClose","ptr",handle)
}


SetUnicodeStr(ByRef out, str_)
{
  VarSetCapacity(out,2*StrPut(str_,"utf-16"))
  StrPut(str_,&out,"utf-16")
}

```



## 屏幕放大
在录制视频上，常需要屏幕放大，按Windows和加号键就可以了，但是会有一个放大镜的图标或窗口浮动在屏幕上，看起来不太好，只需要选中这个窗口，最小化（不是关闭）它就可以了，以后都不会再浮动了。

## autohotkey
使用autohotkey的时候，发现在某些应用下快捷键可用，某些应用下不行，解决方法是设置autohotkey以管理员模式运行就可以了，但是引发下面这个问题。

## 图片批处理
基本上用imagemagik就可以搞定大部分图片批处理：

```
convert src.png -resize 512x taget.png
convert -size 1024x500  radial-gradient:#8c8ca4-#232050 bg.png
convert src.png -crop 1080x1980#0#0 target.png
```

## 管理员模式和自启动
Windows 8不允许自启动列表（startup）中有管理员模式运行的程序，因此上面这个autohotkey问题，如果想开机自启，就做不到了，替代方法是用schedule task代替，创建一个登陆后自启动的任务以替代startup快捷菜单。

## chrome默认雅黑
因为对雅黑偏爱，chrome缺省却不是雅黑，需要在设置里调整standard font， 设置后如下：

![](../../public/images/2018-06-17-07-09-42.png)

## chrome 插件

我使用一些插件来扩展chrome功能，记录一下：
* ImTranslator 用于网页的翻译
* uBlock origin 可以自动屏蔽或移除一些网页的元素，比如在自定义规则中加入 ##.ytp-pause-overlay 可以移除youtube视频pause时的推荐。

## 录音时Mic的噪音
除了软件问题外，Windows级别可以做以下设置，麦克风加强设到最小。

![](../../public/images/2018-06-20-16-52-30.png)

## 电池性能
不需要第三方软件，windows在命令行下输入`powercfg.exe /batteryreport`就可以得到关于电池容量、循环次数、寿命等详细的html报告。

## 显示windows的右下角通知
```
powershell [Reflection.Assembly]::LoadWithPartialName("""System.Windows.Forms""");$obj=New-Object Windows.Forms.NotifyIcon;$obj.Icon = [drawing.icon]::ExtractAssociatedIcon($PSHOME + """\powershell.exe""");$obj.Visible = $True;$obj.ShowBalloonTip(100000, """TITLE""","""NOTIFICATION""",2)>nul
```