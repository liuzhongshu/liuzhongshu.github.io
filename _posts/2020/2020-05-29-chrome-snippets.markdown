---
published: true
title: Chrome的snippets
layout: post
---

Chrome的[Snippets](https://developers.google.com/web/tools/chrome-devtools/javascript/snippets)允许你在任意page上运行js代码。

## 启动
要通过F12或者ctrl-shift-i进入开发者工具，在sources标签页下，左侧找到Sinnpets就是了，看起来挺麻烦，好在下次再通过快捷键进来可以直接定位到这里。

比如给知乎问题排序
```
var sorted = $$(".ListQuestionItem").sort((a,b)=>{
    let aa = $(".QuestionWaiting-info",a).innerText.replace(",","").match(/[0-9]+/g);
    let bb = $(".QuestionWaiting-info",b).innerText.replace(",","").match(/[0-9]+/g);
    return parseInt(aa[1])/parseInt(aa[0]) > parseInt(bb[1])/parseInt(bb[0]) ? -1 : 1;
    });
$(".QuestionWaiting-questions").append(...sorted);
```

