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

## `hello world`和`hello cargo`

### `hello world`

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
