---
layout: post
title: "数据结构与算法 - 第2章: 线性表"
date: 2025-09-12 01:00:00
tags: notes DSA
categories: DSA
---

## 2.1 线性表的概念

### 线性表的定义

- **线性表**：$n$（$n \geq 0$）个数据元素的有序序列
- 表示为二元组：$(K, R)$
  - $K = \{a_0, a_1, \ldots, a_{n-1}\}$：数据元素集合
  - $R = \{r: \text{线性关系}\}$：元素间的关系

### 线性表的特征

#### 结构特点

- **唯一的开始结点**：没有前驱，但有一个唯一的直接后继
- **唯一的终止结点**：没有后继，但有一个唯一的直接前驱
- **内部结点**：有唯一的直接前驱，也有一个唯一的直接后继
- **空表**：长度（包含的结点个数）为0的线性表

#### 关系性质

- 线性表的关系r是**前驱关系**，应具有**反对称性**和**传递性**
- 每个元素都有自己的位置 $[0, n-1]$
- 要求：内部结点具有相同的数据类型

## 2.2 线性表的运算

### 运算分类

#### 构造与析构

- `list()`：创建线性表的一个实例（即构造函数）
- `~list()`：线性表消亡（即析构函数）

#### 信息获取操作

- **位置寻内容，内容找位置**
- `length()`：返回此线性表的当前实际长度
- `isEmpty()`：线性表为空返回true
- `getvalue(const int p, T & value)`：把p位置的值返回到变量value中
- `getPos(int &p, const T value)`：查找值为value的元素，并返回第1次出现的位置

#### 访问和修改操作

- **插入、删除、更改等**
- `clear()`：将线性表存储的内容清除，成为空表
- `append(const T value)`：表尾添加元素value，表长加1
- `insert(const int p, const T value)`：在p位置插入值value，表长加1
- `delete(const int p)`：删去位置p的元素，表长减1
- `setValue(const int p, const T value)`：用value修改位置p的元素值

### 线性表ADT

```cpp
template<class T>
class List { //线性表类模板list，模板参数T
public:
    void clear();                               //置空线性表
    bool isEmpty();                             //线性表为空返回true
    void append(const T value);                 //表尾添加元素value，表长加1
    void insert(int p, T value);               //在p处插入value，表长加1
    void delete(int p);                        //删去第p元素，表长减1
    bool getPos(int &p, T value);              //查找value，并返回其位置
    bool getvalue(const int p, T & value);     //把p位置的值返到value
    bool setvalue(const int p, T & value);     //用value修改p处值
};
```

## 2.3 线性表的存储结构

### 存储方式分类

#### 定长、静态的存储结构

- **又称为向量型的一维数组结构**
- 地址相邻表达线性关系，存储在连续的地址空间，随机访问，但长度固定

#### 变长、动态的存储结构

- **链式存储结构**
  - 指针指向表达线性关系
- **动态数组**
  - 提供空间表管理，为长度变化提供方法，长度增大，可申请大空间

## 2.4 顺序表—向量

### 顺序表的概念

- **顺序表(Sequential list)**，又称**向量(Vector)**
- 采用定长的一维数组存储结构

### 主要特性

- 元素的类型相同
- 存储在连续的空间中，每个元素唯一的索引值（下标），读写元素方便
- 使用常数作为向量长度，程序运行时保持不变

### 逻辑和物理存储结构

#### 地址计算公式

$$\text{Loc}(k_i) = b + L \times i$$

其中：

- **基地址**: $b = \text{Loc}(k_0)$
- **偏移量**: $L = \text{sizeof}(\text{ELEM})$

### 向量的类定义

```cpp
enum Boolean {False,True};
const int Max_length = 100;

template <class T> //假定顺序表的元素类型T
class arrList { //顺序表，向量
private :
    T* aList;        //私有变量，存储顺序表实例的向量
    int maxSize;     //私有变量，顺序表实例的最大长度
    int curLen;      //私有变量，顺序表实例的当前长度
    int position;    //私有变量，当前处理位置
public:
    arrList(const int size);     //构造算子，实参是表实例的最大长度
    ~arrList();                  //析构算子，用于将该表实例删去
    // ... 其他方法
};
```

### 顺序表的基本操作

#### 构造函数和析构函数

```cpp
arrList(const int size) { // 创建一个新顺序表，参数为表实例的最大长度
    maxSize = size;
    aList = new T[maxSize];
    curLen = position = 0;
}

~arrList() { // 析构函数，用于消除该表实例
    delete [] aList;
}

void clear() { // 将顺序表存储的内容清除，成为空表
    delete [] aList;
    curLen = position = 0;
    aList = new T[maxSize];
}
```

#### 查找元素

```cpp
template <class T> // 假定顺序表的元素类型为T
bool arrList<T> :: getPos (int & p, const T value) {
    int i; // 元素下标
    for (i = 0; i < n; i++) // 依次比较
        if (value == aList[i]) { // 下标为i的元素与value相等
            p = i; // 将下标由参数p返回
            return true;
        }
    return false; // 顺序表没有元素值为value的元素
}
```

**时间复杂度**：$O(n)$

#### 插入元素

```cpp
template <class T> // 假定顺序表的元素类型为T
bool arrList<T> :: insert(int p, const T value) {
    int i;
    if (curLen >= maxSize)          // 检查顺序表是否溢出
        return false;
    if (p < 0 || p > curLen)        // 检查插入位置是否合法
        return false;
    for (i = curLen; i > p; i--)
        aList[i] = aList[i-1];      // 从表尾curLen -1起往右移动直到p
    aList[p] = value;               // 位置p处插入新元素
    curLen++;                       // 表的实际长度增1
    return true;
}
```

**插入操作的时间复杂度**：$O(n)$

- 平均移动元素次数为 $\frac{n}{2}$

#### 删除元素

```cpp
template <class T> // 顺序表的元素类型为T
bool arrList<T> :: delete(int p) {
    int i;
    if (curLen <= 0 )               // 检查顺序表是否为空
        return false ;
    if (p < 0 || p > curLen-1)      // 检查删除位置是否合法
        return false ;
    for (i = p; i < curLen-1; i++)
        aList[i] = aList[i+1];      // 从位置p开始每个元素左移直到curLen
    curLen--;                       // 表的实际长度减1
    return true;
}
```

**删除操作的时间复杂度**：$O(n)$

## 2.5 链表(Linked List)

### 链表的概念

- **链表(linked list)**
  - 指针指向保持前驱关系，节点不必物理相邻
  - 动态申请/释放空间，长度动态变化（插入/删除）
- 在非线性结构（如树、图）中的应用

### 链表的分类

- **单链表**
- **双链表**
- **循环链表**

### 单链表

#### 结点类型定义

```cpp
struct ListNode
{
    ELEM data;          //存放线性表结点的数据
    ListNode * next;    //存放指向后继结点的指针
};

typedef ListNode * ListPtr;
ListPtr head, tail;     // head是分别指向单链表头、尾结点的指针
```

#### 单链表头节点

- **Header Node**（或称"哨兵"）
  - 不被作为表中的实际元素，值忽略
  - head指向该节点
- **访问**
  - 必须从head开始查找链表中的元素

#### 设置头结点的好处

1. 由于开始结点的位置被存放在头结点的指针域中，所以在链表的第一个位置上的操作就和在表的其它位置上操作一致，无须进行特殊处理
2. 无论链表是否为空，其头指针是指向头结点的非空指针（空表中头结点的指针域空），因此空表和非空表的处理也就统一了

#### 链表类型定义

```cpp
template <class T> class lnkList : public List<T> {
private:
    Link<T> *head, tail;            // 单链表的头、尾指针
    Link<T> *setPos(int p);         // 返回线性表指向第p个元素的指针值
public:
    lnkList(int s);                 // 构造函数
    ~lnkList();                     // 析构函数
    bool isEmpty();                 // 判断链表是否为空
    void clear();                   // 将链表存储的内容清除，成为空表
    int length();                   // 返回此顺序表的当前实际长度
    bool append(T value);           // 在表尾添加一个元素value，表的长度增1
    bool insert(int p, T value);    // 在位置p插入一个元素value，表的长度增1
    bool delete(int p);             // 删除位置p上的元素，表的长度减 1
    bool getValue(int p, T value);  // 返回位置p的元素值
    bool getPos(int p, const T value); // 查找值为value的元素，并返回第1次出现的位置
}
```

### 链表的基本操作

#### 链表检索

```cpp
// 返回位置i处的结点指针
template <class T> // 线性表的元素类型为T
Link<T> * lnkList <T>:: setPos(int i) {
    int count = 0;
    if (i <= -1) return head;       // i 为-1则定位到头结点
    Link<T> *p = head->next;
    while (p != NULL && count < i) { // 若i为0则定位到第1个结点
        p = p-> next;
        count++;
    };
    return p;                       // 或者为空，或者指向第i个节点！
} // i从0开始！
```

#### 链表插入

```cpp
ListNode * Insert(int i, T value)
{
    ListNode *p, *q;
    q = new ListNode;               //产生一个新结点空间q
    p = setPos(i-1);                //找到待插位置的前一个位置p
    if (p == NULL ) return false;   //位置i无效
    q->data = value;
    q->next = p->next;
    p->next = q;
    if(q->next == NULL )
        tail=q;                     //当插入元素是最后位置时维护尾指针
    return true;
}
```

#### 链表删除

```cpp
template <class T> // 线性表的元素类型为T
bool lnkList<T>:: delete(const int i) {
    Link<T> *p, *d;
    if ((p = setPos(i-1)) == NULL || p == tail) { // 待删结点不存在
        cout << " 非法删除点 " <<endl; return false;
    }
    d = p->next;                    // d是真正待删结点
    if (d == tail) {                // 待删结点为尾结点，则修改尾指针
        tail = p;
        p->next = NULL;
        delete d;
    }
    else { p->next = d->next; delete d; } // 删除结点d并修改链指针
    return true;
}
```

### 双链表

#### 类型说明

```cpp
struct DblListNode
{
    T data;
    DblListNode *prev;      // 指向前驱结点
    DblListNode *next;      // 指向后继结点
};

struct DoubleList
{
    DblListNode * head, *tail;
};
```

#### 删除结点操作

```cpp
// 删除p所指的结点setPos(i)
p->prev->next = p->next;
p->next->prev = p->prev;
p->prev = NULL;
p->next = NULL;
delete(p);
```

#### 插入结点操作

```cpp
// 在p所指结点后插入一个新的结点
new q;                      // ①
q->next = p->next;          // ②
q->prev = p;                // ③
p->next = q;                // ④
q->next->prev = q;
```

**注意**：要注意操作的次序和边界条件的判断！

### 循环链表

- 将单链表或者双链表的头尾结点链接起来，就是一个循环链表
- 不增加额外存储花销，却给不少操作带来了方便
  - 从循环表中任一结点出发，都能访问到表中其他结点

## 2.6 线性表实现方法的比较

### 顺序表的优点

- 没有使用指针，不用花费额外开销
- 线性表元素的访问非常便利
- 读取元素方便，时间代价为O(1)

### 链表的优点

- 无需事先了解线性表的长度
- 允许线性表的长度动态变化
- 能够适应经常插入删除内部元素的情况

### 性能比较

#### 时间复杂度对比

| 操作 | 顺序表 | 链表   |
| ---- | ------ | ------ |
| 查找 | $O(1)$ | $O(n)$ |
| 插入 | $O(n)$ | $O(1)$ |
| 删除 | $O(n)$ | $O(1)$ |

#### 存储开销

- **顺序表**：如果整个数组元素很满，则没有结构性存储开销
- **链表**：每个元素都有结构性存储开销（指针）

### 应用场合的选择

#### 顺序表不适用的场合

- 经常插入删除时，不宜使用顺序表
- 线性表的最大长度也是一个重要因素

#### 链表不适用的场合

- 当读操作比插入删除操作频率大时，不应选择链表
- 当指针的存储开销，和整个结点内容所占空间相比其比例较大时，应该慎重选择

### 选择原则

- **顺序表适合存储静态数据，链表适合动态数据**
- 结点变化的动态性
- 存储密度

## 2.7 约瑟夫问题

### 问题描述

n个人围坐在一圆桌周围，现从第s个人开始报数，数到第m的人出列，然后从出列的下一个人重新开始报数，数到第m的人又出列，如此反复直到所有的人全部出列为止。

**问题描述**：对于任意给定的n, s和m，求按出列次序得到的人员序列。

参数说明：

- **n**: 参与游戏的人数，每个人的信息
- **s**: 开始的人
- **m**: 单次计数

### 解决方案

#### 顺序表方式实现

**步骤**：

1. 建立顺序表
2. 出列算法
3. 主程序

**时间复杂度分析**：

- 运行时间主要是出列元素的删除（数组的移动）
- 每次最多移动$i-1$个元素，总计个数不超过：
  $$(n-1)+(n-2)+\cdots+1 = \frac{n(n-1)}{2} \Rightarrow O(n^2)$$

#### 循环链表方式实现

使用循环链表可以更高效地解决约瑟夫问题，避免频繁的元素移动操作。

## 本章总结

### 核心概念

1. **线性表**：具有线性关系的数据元素的有序序列
2. **存储结构**：顺序存储（数组）和链式存储（指针）
3. **基本操作**：插入、删除、查找、修改

### 重要知识点

1. **顺序表 vs 链表**：

   - 顺序表：随机访问$O(1)$，插入删除$O(n)$
   - 链表：顺序访问$O(n)$，插入删除$O(1)$

2. **头结点的作用**：

   - 统一空表和非空表的处理
   - 简化插入删除操作的边界处理

3. **循环链表的优势**：
   - 从任一结点都能访问到其他所有结点
   - 适合解决约瑟夫问题等循环操作

### 思考问题

- 带表头与不带表头的单链表？
- 处理链表需要注意的问题？
- 线性表实现方法的比较？
- 顺序表和链表的选择依据？
