---
layout: post
title: "数据结构与算法 - 第5章: 二叉树（Binary Tree）"
date: 2025-09-26 14:00:00
tags: notes DSA
categories: DSA
---

## 二叉树的概念

### 定义

**二叉树**（binary tree）由结点的有限集合构成：

- 或者为空集（NIL）
- 或者由一个根结点及两棵不相交的分别称作左子树和右子树的二叉树组成

这是一个**递归定义**。二叉树或为空集，或者空左子树，或者空右子树，或者左右子树皆空。

{% include figure.liquid path="assets/img/notes/dsa/binary-tree-definition.jpg" class="img-fluid rounded z-depth-1" %}

### 五种基本形态

二叉树具有五种基本形态：

1. **空二叉树**
2. **根和空的左、右子树**
3. **根和非空左子树、空右子树**
4. **根和空左子树、非空右子树**
5. **根和非空的左、右子树**

{% include figure.liquid path="assets/img/notes/dsa/binary-tree-forms.jpg" class="img-fluid rounded z-depth-1" %}

### 相关概念

- **父母（parent）**：结点的上层结点
- **子女（孩子）（children）**：结点的下层结点
- **边（edge）**：连接结点的线
- **兄弟（sibling）**：同一父母的子女结点
- **路径（path）**：从一个结点到另一结点的路线
- **祖先（ancestor）**：从根到该结点路径上的所有结点
- **子孙（descendant）**：以某结点为根的子树中的所有结点
- **树叶（leaf）**：度为0的结点
- **内部节点或分支节点（internal node）**：度不为0的结点
- **度数（degree）**：结点子树的数目
- **层数（level）**：根结点层数为0，其它结点层数等于父母层数加1

### 特殊二叉树

#### 满二叉树

如果一棵二叉树的结点，或为树叶（0度节点），或为两棵非空子树（2度节点），则称作满二叉树。

**特点**：1度节点个数为0

{% include figure.liquid path="assets/img/notes/dsa/full-binary-tree.jpg" class="img-fluid rounded z-depth-1" %}

#### 完全二叉树

若一棵二叉树：

- 最多只有最下面的两层结点度数可以小于2
- 最下面一层的结点都集中在该层最左边、连续位置上

则称此二叉树为完全二叉树。

**完全二叉树的特点**：

- 叶结点只可能在最下面两层出现
- 路径长度和最短（满二叉树不具有此性质）
- 由根结点到各个结点的路径长度总和在具有同样结点数的二叉树中最小

{% include figure.liquid path="assets/img/notes/dsa/complete-binary-tree.jpg" class="img-fluid rounded z-depth-1" %}

#### 扩充二叉树

当二叉树节点出现空指针时，就增加一个特殊结点——空树叶：

- 度为1的结点，在它下面增加1个空树叶
- 度为0的树叶，在它下面增加2个空树叶

扩充的二叉树是满二叉树，新增加空树叶（外部结点）的个数等于原来二叉树结点个数（内部结点）加1。

**扩充二叉树性质**：

- **外部路径长度E**：从扩充二叉树的根到每个外部结点的路径长度之和
- **内部路径长度I**：从扩充二叉树的根到每个内部结点的路径长度之和
- **E和I的关系**：$E = I + 2n$，其中n是内部节点的个数

## 二叉树的基本性质

### 性质1

在非空二叉树的第i层上至多有 $2^i$ 个结点（$i \geq 0$）

**证明**：用归纳法

1. $i=0$ 时，二叉树中只有一个根结点，$2^0 = 1$ 成立
2. 假定对所有的 $j (0 \leq j \leq i)$，命题成立，即第j层上至多有 $2^j$ 个结点
3. 第 $i+1$ 层上的最大结点个数是第i层上最大结点个数的2倍，即 $2 \times 2^i = 2^{i+1}$

### 性质2

深度为k的二叉树中最多有 $2^{k+1} - 1$ 个结点（$k \geq 0$）

**证明**：
$$M = \sum_{i=0}^{k} m_i \leq \sum_{i=0}^{k} 2^i = 2^{k+1} - 1$$

### 性质3

对于任何一棵非空的二叉树，如果叶结点个数为 $n_0$，度为2的结点个数为 $n_2$，则有：
$$n_0 = n_2 + 1$$

**证明**：
设二叉树中有n个结点，度为1的结点个数为 $n_1$，则：

- $n = n_0 + n_1 + n_2$ ... (1)
- $B = n - 1$ （B为边的总数）... (2)
- $B = n_1 + 2n_2$ （边都是由度为1和2的结点发出）... (3)

综合(1)、(2)、(3)式可得：$n_0 = n_2 + 1$

### 性质4

具有n个结点的完全二叉树的深度 $k = \lfloor \log_2 n \rfloor$

### 性质5

对于具有n个结点的完全二叉树，如果按照从上到下和从左到右的顺序对树中的所有结点从0开始进行编号，则对于任意的序号为i的结点，有：

1. 如果 $i > 0$，则其父结点的序号为 $\lfloor (i-1)/2 \rfloor$；如果 $i=0$，则其是根结点
2. 如果 $2i+1 \leq n-1$，则其左子女结点的序号为 $2i+1$；否则，其没有左子女结点
3. 如果 $2i+2 \leq n-1$，则其右子女结点的序号为 $2i+2$；否则，其没有右子女结点

### 性质6

在非空满二叉树中，叶节点的个数比分支节点的个数多1。

### 性质7

在扩充的二叉树里，新增加的外部结点的个数比原来的内部结点个数多1。

### 性质8

对任意扩充二叉树，E和I之间满足以下关系：$E = I + 2n$，其中n是内部结点个数。

## 二叉树的遍历

### 遍历定义

**遍历**（Traversal），也称"周游"：

- 按照一定的次序（规律）系统地访问二叉树中的结点
- 每个结点都正好被访问（输出，修改节点信息等）一次

**二叉树的线性化**：

- 实质是把二叉树的结点放入一个线性序列的过程
- "非线性" → "线性" 的过程

### 线性化方式

- **深度优先**：一棵一棵子树的纵深遍历
- **广度优先**：一层一层的自左而右的逐层横向遍历

### 深度优先遍历

变换根结点的周游顺序，可以得到以下六种方案：

#### 前序遍历

- 访问根结点 → 前序遍历左子树 → 前序遍历右子树
- 访问根结点 → 前序遍历右子树 → 前序遍历左子树

#### 中序遍历

- 中序遍历左子树 → 访问根结点 → 中序遍历右子树
- 中序遍历右子树 → 访问根结点 → 中序遍历左子树

#### 后序遍历

- 后序遍历左子树 → 后序遍历右子树 → 访问根结点
- 后序遍历右子树 → 后序遍历左子树 → 访问根结点

特点：

- **根节点遍历时机的决定性**
- **子树遍历结果的连续性**
- **遍历过程的递归性**

### 遍历示例

对于下图的二叉树：

{% include figure.liquid path="assets/img/notes/dsa/binary-tree-traversal-example.jpg" class="img-fluid rounded z-depth-1" %}

- **前序遍历**：ABDGCEFH
- **中序遍历**：DGBAECHF
- **后序遍历**：GDBEHFCA

### 递归实现

```cpp
template<class T>
void BinaryTree<T>::DepthOrder(BinaryTreeNode<T>* root) {
    if (root != NULL) {
        Visit(root);                        // 前序
        DepthOrder(root->leftchild());      // 递归访问左子树
        Visit(root);                        // 中序
        DepthOrder(root->rightchild());     // 递归访问右子树
        Visit(root);                        // 后序
    }
}
```

### 对遍历的分析

从递归遍历算法可知：如果将Visit(root)语句抹去，从递归的角度看，这三种算法是完全相同的，或者说这三种遍历算法的访问路径是相同的，只是访问结点的时机不同。

从虚线的出发点到终点的路径上，每个结点经过3次：

- **第1次经过时访问**，是先序遍历
- **第2次经过时访问**，是中序遍历
- **第3次经过时访问**，是后序遍历

**时间复杂度**：$O(n)$ （每个结点线性访问次数）
**空间复杂度**：$O(n)$ （栈占用的最大可能辅助空间）
精确值：树深为k的递归遍历需要k+1个辅助单元

### 遍历的性质

#### 性质1

已知二叉树的先序序列和中序序列，可以唯一确定一棵二叉树。
推论：已知二叉树的后序序列和中序序列，可以唯一确定一棵二叉树。

#### 性质2

已知二叉树的先序序列和后序序列，不能唯一确定一棵二叉树。

### 已知前序和中序序列求解树的方法

1. **确定树的根节点**：树根是当前树中所有元素在前序序列中的第一个元素
2. **求解树的子树**：找出根节点在中序序列中的位置，根左边的所有元素就是左子树，根右边的所有元素就是右子树
   - 若根节点左边或右边为空，则该方向子树为空
   - 若根节点左边和右边都为空，则根节点为叶节点
3. **递归求解树**：将左、右子树分别看成一棵二叉树，重复上述步骤，直到所有节点完成定位

### 非递归实现

**递归与非递归的关系**：

- 递归带来大量函数调用，有许多额外的时间开销
- 理论上所有的递归都是可以转换成非递归的
- 实现算法的非递归转换，需要借助临时的栈来实现

#### 前序遍历非递归算法

基本思想：

- 看到一个结点，访问他，并把非空右子结点压栈，然后深度遍历其左子树
- 左子树遍历完毕，弹出结点并访问之，继续遍历

#### 中序遍历非递归算法

基本思想：

- 遇到一个结点：入栈，遍历其左子树
- 遍历完左子树：出栈并访问之，遍历右子树

#### 后序遍历非递归算法

基本思想：

- 遇到一个结点，将其入栈，遍历其左子树
- 左子树遍历结束后，还不能马上访问栈顶结点，而是要按照其右链去遍历其右子树
- 右子树遍历后才能从栈顶托出该结点访问之

### 广度优先遍历

从二叉树的根结点开始，自上而下逐层遍历；同层节点，按从左到右的顺序对结点逐一访问。

**算法实现**：使用队列数据结构

```cpp
void BinaryTree<T>::LevelOrder(BinaryTreeNode<T>* root) {
    using std::queue;
    queue<BinaryTreeNode<T>*> aQueue;
    BinaryTreeNode<T>* pointer = root;

    if (pointer) aQueue.push(pointer);

    while (!aQueue.empty()) {
        pointer = aQueue.pop();
        Visit(pointer->value());

        if(pointer->leftchild())
            aQueue.push(pointer->leftchild());
        if(pointer->rightchild())
            aQueue.push(pointer->rightchild());
    }
}
```

**复杂性分析**：

- **时间复杂度**：$O(n)$
- **空间复杂度**：$O(n)$（队列占用的最大可能空间）

## 二叉树的存储结构

### 动态存储结构

#### 二叉链表表示法

- 各结点随机存储在内存空间，结点之间关系用指针表示
- 除存储结点本身数据外，每个结点再设置两个指针字段left和right，分别指向左孩子和右孩子
- 子女为空时指针为空指针

结点形式：

```
| left | info | right |
```

#### 三叉链表

除left和right指针外，每个结点再增加一个指向父节点的指针parent，形成"三叉链表"，提供了"向上"访问的能力。

结点形式：

```
| left | item | Parent | right |
```

### 静态存储结构

#### 顺序存储（完全二叉树）

适用于完全二叉树，利用完全二叉树层次序列的规律性，使用数组进行存储。

**优点**：

- 不需要额外的指针空间
- 可以利用数组下标快速定位父子关系

**缺点**：

- 只适用于完全二叉树
- 对于一般二叉树会浪费大量空间

### ADT实现

#### BinaryTreeNode类

```cpp
template <class T>
class BinaryTreeNode {
    friend class BinaryTree<T>;
private:
    T info;                                     // 二叉树结点数据域
public:
    BinaryTreeNode();                           // 缺省构造函数
    BinaryTreeNode(const T& ele);               // 给定数据的构造
    BinaryTreeNode(const T& ele, BinaryTreeNode<T> *l,
                   BinaryTreeNode<T> *r);       // 子树构造结点

    T value() const;                            // 返回当前结点数据
    BinaryTreeNode<T>* leftchild() const;       // 返回左子树
    BinaryTreeNode<T>* rightchild() const;      // 返回右子树
    void setLeftchild(BinaryTreeNode<T>*);      // 设置左子树
    void setRightchild(BinaryTreeNode<T>*);     // 设置右子树
    void setValue(const T& val);                // 设置数据域
    bool isLeaf() const;                        // 判断是否为叶结点
};
```

#### BinaryTree类

```cpp
template <class T>
class BinaryTree {
private:
    BinaryTreeNode<T>* root;                    // 二叉树根结点
public:
    BinaryTree() {root = NULL;}                 // 构造函数
    ~BinaryTree() {DeleteBinaryTree(root);}     // 析构函数
    bool isEmpty() const;                       // 判定二叉树是否为空树
    BinaryTreeNode<T>* Root() {return root;}    // 返回根结点

    void PreOrder(BinaryTreeNode<T> *root);     // 前序遍历二叉树
    void InOrder(BinaryTreeNode<T> *root);      // 中序遍历二叉树
    void PostOrder(BinaryTreeNode<T> *root);    // 后序遍历二叉树
    void LevelOrder(BinaryTreeNode<T> *root);   // 按层次遍历二叉树
    void DeleteBinaryTree(BinaryTreeNode<T> *root); // 删除二叉树
};
```

---

_本笔记基于北京大学《数据结构与算法》第五章内容整理_
