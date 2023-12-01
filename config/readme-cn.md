## 概念
在处理yaml文件过程中提供了类似面向对象编程的一系列功能，这样可以让你的配置文件更加简洁，同时也可以让你的配置文件更加灵活。下面将介绍如何配置这些功能。

### 继承
继承的关键字为: `parent_path`，它的值为一个字符串，表示继承的父节点的路径。例如：
```yaml
# 我们真实使用的config文件

data1: 'real'
parent_path: parent # 这里为继承的父yaml文件的路径，可以为绝对路径，也可以是相对路径（相对路径是以当前yaml文件所在路径开始算起）
```
```yaml
# 父yaml中的内容

data1: 'parent'
data2: 'parent'
```
那么我们读取后的最终yaml结果为:
```yaml

data1: 'real'
data2: 'parent'
```
目前未提供二级子属性的注入修改

### 引用
引用的关键字为: `definitions`，它的值为一个对象，表示引用的对象yaml路径集合，其后缀需要为`_path`。例如：
```yaml
# 我们真实使用的config文件
definitions:
  key1_path: 'path1.yaml'
```

```yaml
# path1.yaml

data1: 'real'
```

那么我们读取后的最终yaml结果为:
```yaml
key1:
  data1: 'real'
```

注意: 引用是支持级联操作的，例如：
```yaml
# 我们真实使用的config文件
definitions:
  key1_path: 'path1.yaml'
```
```yaml
# path1.yaml
data1: 'real'
definitions:
  key2_path: 'path2.yaml'
```
```yaml
# path2.yaml
data2: 'path2'
```
那么最终读取结果为:
```yaml
key1:
  data1: 'real'
  key2:
    data2: 'path2'
```

引用同样也可以和继承一起使用

## 用法
我们通过继承和引用可以在yaml中简单的实现快速定义相关通用配置的能力。

一个完整的例子可在[config_demo.yaml](config_demo.yaml)中查看。

## 拓展
- [] 支持二级(多)子属性的注入修改
- [] 将yaml_read单独拆分出来，作为一个独立的包引入配置