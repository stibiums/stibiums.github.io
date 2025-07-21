---
layout: post
title: notes of Rustlearning
date: 2025-07-08 08:00:00
description: 这是我学习Rust语言的笔记.
tags: notes
categories: my-notes
toc:
  sidebar: left
---

学习使用的[教程](https://kaisery.github.io/trpl-zh-cn/title-page.html)

## hello world 和 hello cargo

### hello world

创建一个以`hello.rs`文件，内容如下：

```rust
fn main() {
    println!("Hello, world!");
}
```

然后在终端中运行：

```bash
rustc hello.rs
./hello
```

便会运行输出`Hello, world!`。这是最简单编写运行Rust程序的方式。

### cargo

Cargo 是 Rust 的构建系统和包管理器。使用 Cargo 来管理 Rust 项目，它可以为你处理很多任务，比如构建代码、下载依赖库并编译这些库。

使用cargo创建项目的方法：

```bash
cargo new hello_cargo
cd hello_cargo
```

进入 hello_cargo 目录并列出文件。将会看到 Cargo 生成了两个文件和一个目录：一个 Cargo.toml 文件，一个 src 目录，以及位于 src 目录中的 main.rs 文件。

如果要在已有的文件里使用 Cargo，可以在项目目录下运行：

```bash
cargo init
```

### 编译和运行

在项目目录下运行：

```bash
cargo build
```

这会编译项目并生成可执行文件。可执行文件位于 `target/debug/` 目录下。如果需要发布时可以加上参数`--release`，这样会进行优化编译：

要运行项目，可以使用：

```bash
cargo run
```

Cargo 还提供了一个叫 `cargo check` 的命令。该命令快速检查代码确保其可以编译，但并不产生可执行文件

## 常见编程概念的迁移

### 变量和可变性

#### 变量的定义

在 Rust 中，使用`let`声明一个变量。变量默认是不可变的。要声明一个可变变量，可以使用 `mut` 关键字：

```rust
let x = 5; // 不可变变量
let mut y = 10; // 可变变量
```

如果尝试修改不可变变量，会导致编译错误。Rust的编译器会保证不可变的变量不会发生改变。

#### 常量

常量使用 `const` 关键字声明，在声明时也必须指明其数据类型，必须在编译时就确定值。常量可以在任何作用域中使用，包括函数内部和外部。

```rust
const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
```

常量可以用常量表达式来定义

在声明它的作用域之中，常量在整个程序生命周期中都有效

#### 遮蔽

遮蔽是指使用同名变量来隐藏之前的变量。可以通过重新声明一个变量来遮蔽之前的变量。

第二个变量遮蔽了第一个变量，此时任何使用该变量名的行为中都会视为是在使用第二个变量，直到第二个变量自己也被遮蔽或第二个变量的作用域结束。

```rust
fn main() {
    let x = 5;

    let x = x + 1;

    {
        let x = x * 2;
        println!("The value of x in the inner scope is: {x}");
    }

    println!("The value of x is: {x}");
}
```

遮蔽和变量修改是有不同的；遮蔽创建的新的变量，修改原来的变量则是改变原来的变量的值。

### 数据类型

Rust 是一种静态类型的语言，在编译时就必须确定所有变量的类型。在定义变量或者常量时，可以使用类型注解来指定变量的类型。

有两种数据类型：标量类型和复合类型。

#### 标量类型

标量类型表示单一的值。Rust 中有四种基本的标量类型：整数、浮点数、布尔值和字符。

##### 整型

以下是 Rust 中整型的简洁表格：

| **类型** | **长度** | **范围**                        |
| -------- | -------- | ------------------------------- |
| `i8`     | 8 位     | -2⁷ 到 2⁷-1                     |
| `u8`     | 8 位     | 0 到 2⁸-1                       |
| `i16`    | 16 位    | -2¹⁵ 到 2¹⁵-1                   |
| `u16`    | 16 位    | 0 到 2¹⁶-1                      |
| `i32`    | 32 位    | -2³¹ 到 2³¹-1                   |
| `u32`    | 32 位    | 0 到 2³²-1                      |
| `i64`    | 64 位    | -2⁶³ 到 2⁶³-1                   |
| `u64`    | 64 位    | 0 到 2⁶⁴-1                      |
| `isize`  | 平台相关 | 取决于操作系统（32 位或 64 位） |
| `usize`  | 平台相关 | 取决于操作系统（32 位或 64 位） |

---

**说明**：

1. **`i` 表示有符号整型**，可以表示正数和负数。
2. **`u` 表示无符号整型**，只能表示非负数。
3. **`isize` 和 `usize`** 是平台相关的整型类型：
   - 在 32 位系统上，它们分别是 32 位。
   - 在 64 位系统上，它们分别是 64 位。
4. **默认类型**：如果没有显式指定类型，Rust 会默认使用 `i32` 类型。

##### 浮点型

Rust 的浮点数类型是 f32 和 f64，分别占 32 位和 64 位。默认类型是 f64

##### 布尔型

布尔类型只有两个值：true 和 false。可以使用 `bool` 类型来声明

##### 字符型

字符类型使用 `char` 类型来表示，表示单个 Unicode 字符。字符类型是四个字节（32 位），可以表示任何有效的 Unicode 字符，带变音符号的字母（Accented letters），中文、日文、韩文等字符，emoji（绘文字）以及零长度的空白字符都是有效的 char 值

```rust
fn main() {
    let c = 'z';
    let z: char = 'ℤ'; // with explicit type annotation
    let heart_eyed_cat = '😻';
}
```

#### 复合类型

复合类型可以将多个值组合成一个值。Rust 中有两种基本的复合类型：元组和数组。

##### 元组

元组是将多个值组合成一个复合类型。元组的元素可以是不同类型的。元组使用小括号 `()` 来表示，元素之间用逗号分隔。

```rust
fn main() {
    let tup: (i32, f64, u8) = (500, 6.4, 1);

    let (x, y, z) = tup; // 支持解构

    println!("The value of y is: {y}");
}
```

可以使用`.` 加上索引来访问元组的元素

##### 数组

数组是固定长度的同类型元素的集合。数组使用方括号 `[]` 来表示，元素之间用逗号分隔。

```rust
fn main() {
    let a: [i32; 5] = [1, 2, 3, 4, 5]; //方括号中包含每个元素的类型，后跟分号，再后跟数组元素的数量
    let a = [3; 5]; // 创建一个包含 5 个元素的数组，每个元素的值都是 3
    let first = a[0]; // 访问数组的第一个元素
}
```

### 函数

#### 定义

在 Rust 中通过输入 fn 后面跟着函数名和一对圆括号来定义函数。Rust 不关心函数定义所在的位置，只要函数被调用时出现在调用之处可见的作用域内就行。

在函数的签名中，必须声明每一个参数的类型。

```rust
fn print_labeled_measurement(value: i32, unit_label: char) {
    println!("The measurement is: {value}{unit_label}");
}
```

#### 语句和表达式

- **语句**：是执行一些操作但不返回值的指令。以分号结尾
- **表达式**：计算并产生一个值，可以嵌套。不会以分号结尾

```rust
fn main() {
    let y = {
        let x = 3;
        x + 1
    };

    println!("The value of y is: {y}");
}
```

#### 函数返回值

函数可以返回值，使用 `->` 语法来指定返回值的类型。函数的返回值等同于函数体最后一个表达式的值。

```rust
fn five() -> i32 {
    5 // 注意这里没有分号，因为这是一个表达式
}
```

### 控制流

#### `if`

`if` 表达式用于条件判断。可以使用 `else if` 和 `else` 来处理其他情况。

```rust
fn main() {
    let number = 6;
    if number % 4 == 0 {
        println!("Number is divisible by 4");
    } else if number % 3 == 0 {
        println!("Number is divisible by 3");
    } else {
        println!("Number is not divisible by 4 or 3");
    }
}
```

注意的是代码中的条件必须是 bool 值。如果条件不是 bool 值，将会报错。Rust并不会尝试将其他类型转换为布尔值。

因为`if`是一个表达式，所以可以将其结果赋值给变量。

```rust
let condition = true;
let number = if condition { 5 } else { 6 }; // 注意这里的 else分支也必须返回相同类型的值（Rust需要在编译时就明确变量的类型）
println!("The value of number is: {number}");
```

#### `loop`

`loop` 用于创建无限循环。可以使用 `break` 退出循环，使用 `continue` 跳过当前迭代。

`break` 可以返回一个值作为循环的结果。

```rust
fn main() {
    let mut counter = 0;

    let result = loop {
        counter += 1;

        if counter == 10 {
            break counter * 2;
        }
    };

    println!("The result is {result}");
}
```

可以利用循环标签来标记循环，以便在嵌套循环中使用 `break` 或 `continue` 时指定要跳出或继续的循环。

```rust
fn main() {
    'outer: loop {
        println!("Entered the outer loop");
        loop {
            println!("Entered the inner loop");
            break 'outer; // 跳出外层循环
        }
        println!("This line will not be printed");
    }
    println!("Exited the outer loop");
}
```

#### `while`

`while` 循环在条件为 `true` 时执行。可以在循环体中使用 `break` 和 `continue` 来控制循环。

#### `for`

`for` 循环用于遍历集合或范围。可以使用 `in` 关键字来指定要遍历的集合或范围。

```rust
fn main() {
    let a = [10, 20, 30, 40, 50];

    for element in a {
        println!("the value is: {element}");
    }
}
```

## 所有权

**所有权**是 Rust 的核心概念之一。Rust 通过所有权系统来管理内存，确保内存安全和防止数据竞争。

- rust 中的每个值都有一个所有者（变量）。
- 每个值只能有一个所有者。
- 当所有者离开作用域时，值会被自动释放。

### 变量的作用域
