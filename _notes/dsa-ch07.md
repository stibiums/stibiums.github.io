---
layout: post
title: "数据结构与算法 - 第7章: 图（Graph）"
date: 2025-10-25 16:00:00
tags: notes DSA graph
categories: DSA
---

## 7.1 图的基本概念

### 7.1.1 图的定义

**图**（Graph）是一种重要的非线性数据结构，用$$G = (V, E)$$表示：

- $$V$$表示**有限的顶点集合**（Vertex Set）
- $$E$$表示**边集合**（Edge Set），是顶点的偶对$$(边的始点, 边的终点)$$
- $$|V|$$表示顶点的总数，$$|E|$$表示边的总数

图广泛应用于现实世界的建模，如社交网络、交通网络、通信网络等。

### 7.1.2 图的分类

根据不同的特征，图可以进行多种分类：

**按边的数量分类：**

- **稀疏图**（Sparse Graph）：边数相对较少的图
- **密集图**（Dense Graph）：边数相对较多的图
- **完全图**（Complete Graph）：包括所有可能边的图
  - 具有$$n$$个顶点的无向完全图有$$\frac{n(n-1)}{2}$$条边
  - 具有$$n$$个顶点的有向完全图有$$n(n-1)$$条边

**按边的方向分类：**

- **无向图**（Undirected Graph）：顶点偶对无序的图，边用圆括号表示$$(V_i, V_j)$$
- **有向图**（Directed Graph）：顶点偶对有序的图，边用尖括号表示$$\langle V_i, V_j \rangle$$

**按顶点和边的标注分类：**

- **标号图**（Labeled Graph）：各顶点均带有标号的图
- **带权图**（Weighted Graph）：边上标有权值的图

### 7.1.3 图的集合表示

**无向图的集合表示：**

对于无向图$$G_1 = (V, E)$$：

- $$V(G_1) = \{V_1, V_2, V_3, V_4\}$$
- $$E(G_1) = \{(V_1, V_2), (V_1, V_3), (V_1, V_4), (V_2, V_3), (V_2, V_4), (V_3, V_4)\}$$

注意：无向图的边用**圆括号**表示，且$$(V_i, V_j) = (V_j, V_i)$$。

**有向图的集合表示：**

对于有向图$$G_2 = (V, E)$$：

- $$V(G_2) = \{V_1, V_2, V_3\}$$
- $$E(G_2) = \{\langle V_1, V_2 \rangle, \langle V_2, V_1 \rangle, \langle V_2, V_3 \rangle\}$$

注意：有向图的边用**尖括号**表示，且$$\langle V_i, V_j \rangle \neq \langle V_j, V_i \rangle$$。

**本章约定：**

- 不考虑顶点到自身的边（无自环）
- 不允许一条边在图中重复出现（无重边）

### 7.1.4 图的相关概念

**邻接点（Neighbors）：**

一条边所连接的两个顶点称为**邻接点**或**相邻顶点**，这条边称为**相关联的边**。

**顶点的度（Degree）：**

定义为与该顶点相关联边的数目。

对于**有向图**$$G = (V, E)$$而言：

- **入度**（In Degree）：以顶点$$V$$为终点的边的数目
- **出度**（Out Degree）：以顶点$$V$$为始点的边的数目
- **终端结点**（叶子）：出度为0的顶点

**重要性质：**

若图$$G$$有$$n$$个顶点，$$e$$条边，$$d_i$$为顶点$$V_i$$的度数，则：

$$\sum_{i=1}^{n} d_i = 2e$$

这说明**所有顶点的度数之和等于边数的两倍**。

**子图（Subgraph）：**

图$$G = (V, E)$$，$$G' = (V', E')$$中，若$$V' \subseteq V$$，$$E' \subseteq E$$，并且$$E'$$中的边所关联的顶点都在$$V'$$中，则称图$$G'$$是图$$G$$的**子图**。

### 7.1.5 图的路径

**路径（Path）：**

在图$$G = (V, E)$$中，如果存在顶点序列$$V_p, V_{i1}, V_{i2}, \ldots, V_{in}, V_q$$，使得：

- 对于无向图：$$(V_p, V_{i1}), (V_{i1}, V_{i2}), \ldots, (V_{in}, V_q)$$都在$$E$$中
- 对于有向图：$$\langle V_p, V_{i1} \rangle, \langle V_{i1}, V_{i2} \rangle, \ldots, \langle V_{in}, V_q \rangle$$都在$$E$$中

则称从顶点$$V_p$$到顶点$$V_q$$存在一条**路径**。

**简单路径（Simple Path）：**

路径上除了$$V_p$$和$$V_q$$可以相同外，其它顶点都不相同。

**路径长度：**

路径上边的条数。

**回路/环（Cycle）：**

路径上某个顶点与自身连接。

**简单回路：**

首尾顶点相同的简单路径。

**无环图（Acyclic Graph）：**

不包含回路的图。

**有向无环图（Directed Acyclic Graph，DAG）：**

有向的无环图，在拓扑排序等问题中具有重要应用。

### 7.1.6 图的连通性

**有根图：**

一个有向图中，若存在一个顶点$$V_0$$，从此顶点有路径可以到达图中其它所有顶点，则称此有向图为**有根图**，$$V_0$$称作图的**根**。

**连通图（Connected Graph）：**

对无向图$$G = (V, E)$$而言，如果从$$V_1$$到$$V_2$$有一条路径（从$$V_2$$到$$V_1$$也一定有一条路径），则称$$V_1$$和$$V_2$$是**连通的**。若图$$G$$中任意两个顶点都是连通的，则无向图$$G$$是**连通的**。

**连通分量（Connected Component）：**

指无向图的**最大连通子图**。

**强连通图（Strongly Connected Graph）：**

对有向图$$G = (V, E)$$而言，若对于$$G$$中任意两个顶点$$V_i$$和$$V_j$$（$$V_i \neq V_j$$），都有一条从$$V_i$$到$$V_j$$的有向路径，同时还有一条从$$V_j$$到$$V_i$$的有向路径，则称有向图$$G$$是**强连通的**。

**强连通分量：**

有向图强连通的**最大子图**。

**网络（Network）：**

带权的连通图。

**自由树（Free Tree）：**

不带有简单回路的无向图，它是连通的，且具有$$|V| - 1$$条边。

## 7.2 图的抽象数据类型

图的抽象数据类型（ADT）定义了图的基本操作接口：

```cpp
class Graph {                            // 图的ADT
public:
    int VerticesNum();                   // 返回图的顶点个数
    int EdgesNum();                      // 返回图的边数

    // 返回与顶点oneVertex相关联的第一条边
    Edge FirstEdge(int oneVertex);

    // 返回与边PreEdge有相同关联顶点oneVertex的下一条边
    Edge NextEdge(Edge preEdge);

    // 添加一条边
    bool setEdge(int fromVertex, int toVertex, int weight);

    // 删除一条边
    bool delEdge(int fromVertex, int toVertex);

    // 如果oneEdge是边则返回TRUE，否则返回FALSE
    bool IsEdge(Edge oneEdge);

    // 返回边oneEdge的始点
    int FromVertex(Edge oneEdge);

    // 返回边oneEdge的终点
    int ToVertex(Edge oneEdge);

    // 返回边oneEdge的权
    int Weight(Edge oneEdge);
};
```

**边的基类定义：**

```cpp
class Edge {                             // 边的基类
public:
    int from, to, weight;

    Edge() {                             // 构造函数
        from = -1;
        to = -1;
        weight = 0;
    }

    Edge(int f, int t, int w) {          // 构造函数
        from = f;
        to = t;
        weight = w;
    }

    bool operator > (Edge oneEdge) {     // 定义边比较运算符">"
        return weight > oneEdge.weight;
    }

    bool operator < (Edge oneEdge) {     // 定义边比较运算符"<"
        return weight < oneEdge.weight;
    }
};
```

## 7.3 图的存储结构

图的存储结构主要有三种：邻接矩阵、邻接表和十字链表。

### 7.3.1 邻接矩阵（Adjacency Matrix）

**定义：**

**邻接矩阵**是表示顶点间相邻关系的矩阵。若$$G$$是一个具有$$n$$个顶点的图，则$$G$$的邻接矩阵是如下定义的$$n \times n$$矩阵：

$$
A[i, j] =
\begin{cases}
1, & \text{若}(V_i, V_j)\text{或}\langle V_i, V_j \rangle\text{是图的边} \\
0, & \text{若}(V_i, V_j)\text{或}\langle V_i, V_j \rangle\text{不是图的边}
\end{cases}
$$

**加权矩阵：**

对于带权图，邻接矩阵存储边的权值：

$$
A[i, j] =
\begin{cases}
w_{ij}, & \text{若}(V_i, V_j)\text{或}\langle V_i, V_j \rangle\text{是图的边} \\
0\text{或}\infty, & \text{若}(V_i, V_j)\text{或}\langle V_i, V_j \rangle\text{不是图的边}
\end{cases}
$$

**性质分析：**

**无向图的邻接矩阵性质：**

1. 矩阵对称：$$A[i, j] = A[j, i]$$
2. 第$$i$$行或第$$i$$列中1的个数为顶点$$i$$的度
3. 矩阵中1的个数的一半为图中边的数目
4. 容易判断顶点$$i$$和顶点$$j$$之间是否有边相连

**有向图的邻接矩阵性质：**

1. 矩阵不一定对称
2. 第$$i$$行中1的个数为顶点$$i$$的**出度**
3. 第$$i$$列中1的个数为顶点$$i$$的**入度**
4. 矩阵中1的个数为图的边数
5. 容易判断顶点$$i$$和顶点$$j$$是否有边相连

**空间复杂度：**

邻接矩阵的空间代价为$$O(n^2)$$，与边数无关。对于稀疏图（边数远小于$$n^2$$），邻接矩阵造成空间浪费。

**稀疏因子：**

在$$m \times n$$的矩阵中，有$$t$$个非零元素，则稀疏因子$$\delta$$为：

$$\delta = \frac{t}{m \times n}$$

若$$\delta < 0.05$$，可认为是稀疏矩阵。

**基于邻接矩阵的类表示：**

```cpp
class Graphm : public Graph {
private:
    int **matrix;

public:
    Graphm(int numVert) : Graph(numVert) {
        matrix = (int **) new int*[numVertex];
        for (int i = 0; i < numVertex; i++)
            matrix[i] = new int[numVertex];
        for (int i = 0; i < numVertex; i++)
            for (int j = 0; j < numVertex; j++)
                matrix[i][j] = 0;
    }

    ~Graphm() {
        for (int i = 0; i < numVertex; i++)
            delete[] matrix[i];
        delete[] matrix;
    }

    Edge FirstEdge(int oneVertex) {
        Edge myEdge;
        myEdge.from = oneVertex;
        myEdge.to = -1;
        for (int i = 0; i < numVertex; i++) {
            if (matrix[oneVertex][i] != 0) {
                myEdge.to = i;
                myEdge.weight = matrix[oneVertex][i];
                break;
            }
        }
        return myEdge;
    }

    Edge NextEdge(Edge preEdge) {
        Edge myEdge;
        myEdge.from = preEdge.from;
        myEdge.to = -1;
        for (int i = preEdge.to + 1; i < numVertex; i++) {
            if (matrix[preEdge.from][i] != 0) {
                myEdge.to = i;
                myEdge.weight = matrix[preEdge.from][i];
                break;
            }
        }
        return myEdge;
    }

    void setEdge(int from, int to, int weight) {
        if (matrix[from][to] <= 0) {
            numEdge++;
            indegree[to]++;
        }
        matrix[from][to] = weight;
    }

    void delEdge(int from, int to) {
        if (matrix[from][to] > 0) {
            numEdge--;
            indegree[to]--;
        }
        matrix[from][to] = 0;
    }
};
```

### 7.3.2 邻接表（Adjacency List）

**设计思想：**

邻接矩阵的空间代价只与顶点个数$$n$$有关（$$n^2$$），与边无关。对于稀疏图，大量的边不存在，造成空间浪费。**邻接表**既与顶点有关，又与边有关，适合稀疏图的存储。

**结构组成：**

邻接表由两部分组成：

1. **顶点表**：对应$$n$$个顶点，包括顶点数据和指向边表的指针
2. **边链表**：对应$$m$$条边，包括顶点序号和指向边表下一表目的指针

**结点结构：**

```
顶点结点：[data | firstarc]
边（或弧）结点：[adjvex | nextarc | info]
```

其中：

- `data`：顶点数据
- `firstarc`：指向第一条关联边的指针
- `adjvex`：邻接顶点的序号
- `nextarc`：指向下一条边的指针
- `info`：边的权值等信息

**空间代价：**

- $$n$$个顶点$$e$$条边的**无向图**需要$$(n + 2e)$$个存储单元
- $$n$$个顶点$$e$$条边的**有向图**需要$$(n + e)$$个存储单元

当边数$$e$$很小时，可以节省大量的存储空间。

**注意事项：**

- 边表中表目顺序往往按照顶点编号从小到大排列
- 对于有向图，可以保存**出边表**或**入边表**之一即可

**基于邻接表的关键操作实现：**

```cpp
Edge FirstEdge(int oneVertex) {
    Edge myEdge;
    myEdge.from = oneVertex;
    myEdge.to = -1;
    Link<listUnit> *temp = graList[oneVertex].head;
    if (temp->next != NULL) {
        myEdge.to = temp->next->element.vertex;
        myEdge.weight = temp->next->element.weight;
    }
    return myEdge;
}

Edge NextEdge(Edge preEdge) {
    Edge myEdge;
    myEdge.from = preEdge.from;
    myEdge.to = -1;
    Link<listUnit> *temp = graList[preEdge.from].head;
    while (temp->next != NULL && temp->next->element.vertex <= preEdge.to)
        temp = temp->next;
    if (temp->next != NULL) {
        myEdge.to = temp->next->element.vertex;
        myEdge.weight = temp->next->element.weight;
    }
    return myEdge;
}

void setEdge(int from, int to, int weight) {
    Link<listUnit> *temp = graList[from].head;
    while (temp->next != NULL && temp->next->element.vertex < to)
        temp = temp->next;

    if (temp->next == NULL) {
        temp->next = new Link<listUnit>;
        temp->next->element.vertex = to;
        temp->next->element.weight = weight;
        numEdge++;
        indegree[to]++;
        return;
    }

    if (temp->next->element.vertex == to) {
        temp->next->element.weight = weight;
        return;
    }

    if (temp->next->element.vertex > to) {
        Link<listUnit> *other = temp->next;
        temp->next = new Link<listUnit>;
        temp->next->element.vertex = to;
        temp->next->element.weight = weight;
        temp->next->next = other;
        numEdge++;
        indegree[to]++;
        return;
    }
}
```

### 7.3.3 十字链表（Orthogonal List）

**定义：**

十字链表是另一种链式存储结构，可看成是**邻接表和逆邻接表的结合**。

**结点结构：**

**顶点表结点**：对应图的顶点，由3个域组成：

- `data`域：顶点数据
- `firstinarc`指针：指向第一条以该顶点为终点的边
- `firstoutarc`指针：指向第一条以该顶点为起点的边

**边链表结点**：对应有向图的每一条边，共5个域：

- `fromvex`：起点的顶点序号
- `tovex`：终点的顶点序号
- `info`：边权值的信息域
- `fromnextarc`指针：指向下一条以`fromvex`为起点的边
- `tonextarc`指针：指向下一条以`tovex`为终点的边

**名称由来：**

十字链表有两组链表组成：

- 行和列的指针序列
- 每个结点都包含两个指针：同一行的后继，同一列的后继

这种结构类似于稀疏矩阵的十字链表表示，因此得名。

**应用场景：**

十字链表特别适合需要同时高效访问入边和出边的场景，如有向图的遍历和路径搜索。

## 7.4 图的遍历

### 7.4.1 图遍历的基本概念

**图遍历（Graph Traversal）：**

给定图$$G$$和任一顶点$$V_0$$，从$$V_0$$出发系统地访问$$G$$中所有的顶点，每个顶点访问一次，称为**图遍历**。

**需要考虑的问题：**

1. **非连通图**：从一顶点出发，可能不能到达所有其它的顶点
2. **存在回路的图**：也有可能会陷入死循环

**解决办法：**

- 顶点保留一标志位，初始时标志位置未访问（`UNVISITED`）
- 在遍历过程中，当顶点被访问时，标志位置已访问（`VISITED`）

**遍历算法框架：**

```cpp
void graph_traverse(Graph& G) {
    for (int i = 0; i < G.VerticesNum(); i++)
        G.Mark[i] = UNVISITED;

    for (int i = 0; i < G.VerticesNum(); i++)
        if (G.Mark[i] == UNVISITED)
            do_traverse(G, i);
}
```

**重要性：**

遍历是求解图的连通性、拓扑排序和关键路径等问题的基础。

**两类主要方式：**

1. **深度优先搜索**（Depth-First Search, DFS）
2. **广度优先搜索**（Breadth-First Search, BFS）

### 7.4.2 深度优先搜索（DFS）

**基本思想：**

1. 选取一个未访问的点$$V_0$$作为源点
2. 访问顶点$$V_0$$
3. 递归地深搜遍历$$V_0$$邻接到的其他顶点
4. 重复上述过程直至从$$V_0$$有路径可达的顶点都已被访问过
5. 再选取其他未访问顶点作为源点做深搜，直到图的所有顶点都被访问过

**深度优先搜索树（Depth-First Search Tree）：**

遍历过程中形成的树结构，记录了访问顺序。

**算法实现：**

```cpp
void DFS(Graph& G, int V) {
    Visit(G, V);                    // 访问V
    G.Mark[V] = VISITED;            // 标记其标志位

    for (Edge e = G.FirstEdge(V); G.IsEdge(e); e = G.NextEdge(e)) {
        // 递归地按照深度优先的方式访问V邻接的未被访问的顶点
        if (G.Mark[G.ToVertex(e)] == UNVISITED)
            DFS(G, G.ToVertex(e));
    }
}
```

**复杂性分析：**

**时间复杂度：**

- DFS对每一条边处理一次（无向图的每条边从两个方向处理），每个顶点访问一次
- **采用邻接表表示时**：有向图总代价为$$\Theta(|V| + |E|)$$，无向图为$$\Theta(|V| + 2|E|)$$
- **采用邻接矩阵表示时**：处理所有的边需要$$\Theta(|V|^2)$$的时间，所以总代价为$$\Theta(|V| + |V|^2) = \Theta(|V|^2)$$

### 7.4.3 广度优先搜索（BFS）

**基本思想：**

1. 访问顶点$$V_0$$
2. 然后访问$$V_0$$邻接到的所有未被访问过的邻居顶点$$V_{01}, V_{02}, \ldots, V_{0i}$$
3. 再依次访问$$V_{01}, V_{02}, \ldots, V_{0i}$$邻接到的所有未被访问的邻居顶点
4. 如此进行下去，直到访问遍所有的顶点

**广度优先搜索树（Breadth-First Search Tree）：**

遍历过程中形成的树结构，记录了访问顺序。

**算法实现：**

```cpp
void BFS(Graph& G, int V) {
    using std::queue;
    queue<int> Q;

    G.Mark[V] = VISITED;
    Visit(G, V);
    Q.push(V);

    while (!Q.empty()) {
        int V = Q.front();
        Q.pop();

        // 将与该点相邻的每一个未访问点都入队
        for (Edge e = G.FirstEdge(V); G.IsEdge(e); e = G.NextEdge(e)) {
            if (G.Mark[G.ToVertex(e)] == UNVISITED) {
                G.Mark[G.ToVertex(e)] = VISITED;
                Visit(G, G.ToVertex(e));
                Q.push(G.ToVertex(e));
            }
        }
    }
}
```

**复杂度分析：**

广度优先搜索实质上与深度优先相同，只是访问顺序不同而已。二者**时间复杂度也相同**。

### 7.4.4 拓扑排序（Topological Sort）

**问题定义：**

**先决条件**：是指以某种线性顺序来组织多项任务，以便能够在满足先决条件的情况下逐个完成各项任务。**有向无环图**能够模拟先决条件。

**应用实例：**

课程安排问题——学生需要按照先修课程的要求安排学习顺序。

| 先修课程 | 课程代号 | 课程名称   |
| -------- | -------- | ---------- |
| -        | C1       | 高等数学   |
| -        | C2       | 程序设计   |
| C1, C2   | C3       | 离散数学   |
| C2, C3   | C4       | 数据结构   |
| C2       | C5       | 算法语言   |
| C4, C5   | C6       | 编译技术   |
| C4, C9   | C7       | 操作系统   |
| C1       | C8       | 普通物理   |
| C8       | C9       | 计算机原理 |

**拓扑排序（Topological Sort）：**

将一个**有向无环图**中所有顶点在不违反先决条件关系的前提下排成线性序列的过程称为拓扑排序。

对一个有向无环图$$G$$进行拓扑排序，是将$$G$$中所有顶点排成一个线性序列，使得图中任意一对顶点$$u$$和$$v$$，若$$\langle u, v \rangle \in E(G)$$，则$$u$$在线性序列中出现在$$v$$之前。

**拓扑序列（Topological Sequence）：**

拓扑排序形成的序列。

**重要性质：**

**性质1：** 若将图中顶点按拓扑次序排成一行，则图中所有的有向边均是从左指向右的。拓扑序列**不唯一**。

**性质2：** 环存在时不存在拓扑序列。

**拓扑排序的基本思想：**

限定是有向无环图，拓扑排序方法：

1. 从图中选择一个**入度为0**的顶点并输出
2. 从图中删掉此顶点及其所有的出边（出边关联顶点的入度减1）
3. 回到第（1）步继续执行

**环路存在时：**

排序结束，仍有顶点没有被输出，但在剩下的图中找不到入度为0的顶点。

**实现方法：**

1. **基于邻接矩阵的实现**
2. **基于邻接表的实现**
   - 广度优先排序（BFS-TopSort）
   - 深度优先排序（DFS-TopSort）

**BFS方法（基于邻接表）：**

为每个顶点设置一个表示该结点入度字段（`indegree`），入度表：

- 不用检查$$n \times n$$的矩阵
- 直接检查数组就可确定入度为0的顶点

**BFS-TopSort算法：**

```cpp
void TopsortbyQueue(Graph& G) {
    for (int i = 0; i < G.VerticesNum(); i++)
        G.Mark[i] = UNVISITED;

    using std::queue;
    queue<int> Q;

    for (int i = 0; i < G.VerticesNum(); i++) {
        if (G.Indegree[i] == 0)
            Q.push(i);
    }

    while (!Q.empty()) {
        int V = Q.front();
        Q.pop();
        Visit(G, V);
        G.Mark[V] = VISITED;

        for (Edge e = G.FirstEdge(V); G.IsEdge(e); e = G.NextEdge(e)) {
            G.Indegree[G.ToVertex(e)]--;
            if (G.Indegree[G.ToVertex(e)] == 0)
                Q.push(G.ToVertex(e));
        }
    }

    for (int i = 0; i < G.VerticesNum(); i++) {
        if (G.Mark[i] == UNVISITED) {
            Print("图有环");
            break;
        }
    }
}
```

**注意：** 广度优先排序可以判定有环存在。在有环的情况下会提前退出，从而可能没处理完所有的边和顶点。

**DFS方法（基于邻接表）：**

使用栈，得到**逆序序列**。

**DFS-TopSort算法：**

```cpp
void TopsortbyDFS(Graph& G) {
    for (int i = 0; i < G.VerticesNum(); i++)
        G.Mark[i] = UNVISITED;

    int *result = new int[G.VerticesNum()];
    int tag = 0;

    for (int i = 0; i < G.VerticesNum(); i++)
        if (G.Mark[i] == UNVISITED)
            Do_topsort(G, i, result, tag);

    for (int i = G.VerticesNum() - 1; i >= 0; i--) {
        Visit(G, result[i]);
    }
}

void Do_topsort(Graph& G, int V, int *result, int& tag) {
    G.Mark[V] = VISITED;

    for (Edge e = G.FirstEdge(V); G.IsEdge(e); e = G.NextEdge(e))
        if (G.Mark[G.ToVertex(e)] == UNVISITED)
            Do_topsort(G, G.ToVertex(e), result, tag);

    result[tag++] = V;
}
```

**注意：** 深度优先拓扑排序**不能判断环的存在**。

**复杂性分析：**

- **采用邻接矩阵时**：每次算法需要找所有入度为0的顶点，需要$$\Theta(|V|^2)$$的时间，那么对$$|V|$$个顶点而言，总代价为$$\Theta(|V|^3)$$
- **采用邻接表时**：因为在顶点表的每个顶点中可以有一个字段来存储入度，所以只需$$\Theta(|V|)$$的时间，加上处理边、顶点的时间，总代价为$$\Theta(2|V| + |E|)$$

## 7.5 最短路径问题

### 7.5.1 问题定义

**带权图的最短路径问题：**

即求两个顶点间长度最短的路径。其中：路径长度不是指路径上边数的总和，而是指路径上各边的**权值总和**。

**应用场景：**

管线铺设、出行线路选择等。

**注意：** 广度优先遍历本质上就是单位权重图的最短路径搜索问题。

**最短路径问题求解分类：**

1. **单源最短路径**

   - 对已知图$$G = (V, E)$$，给定源顶点$$s \in V$$，找出$$s$$到图中其它各顶点的最短路径
   - 代表性算法：**Dijkstra算法**（贪心思路）

2. **每对顶点间的最短路径**
   - 对已知图$$G = (V, E)$$，任意的顶点$$V_i, V_j \in V$$，找出从$$V_i$$到$$V_j$$的最短路径
   - 代表性算法：**Floyd算法**（动态规划思路）

### 7.5.2 Dijkstra算法

**Edsger Wybe Dijkstra（1930/5/11 ~ 2002/8/6）：**

荷兰人，20世纪最伟大的计算机科学家之一，曾获1972年图灵奖，与D. E. Knuth并称为20世纪最伟大的计算机科学家。

**主要贡献：**

1. 提出"goto有害论"
2. 提出信号量和PV原语
3. 解决了有趣的"哲学家聚餐"问题
4. 最短路径算法（SPF）的创造者
5. 第一个Algol 60编译器的设计者和实现者
6. THE操作系统的设计者和开发者

**Dijkstra算法：**

Dijkstra算法是E. W. Dijkstra于1959年提出的，是目前公认的对**边权非负**情况下的最好算法。

**基本思想：**

每次从距离已生成最短路径的节点集"一步之遥"的节点中，选择距离原点$$V_0$$最近的边进行延伸。结果由近及远生成以起始点$$V_0$$为根的有向树。是一类**贪心算法**。

**路径长度递增序：**

按路径长度递增序产生各顶点最短路径。若按长度递增的次序生成从源点$$s$$到其它顶点的最短路径，则当前正在生成的最短路径上除终点以外，其余顶点的最短路径均已生成。

**实现策略：**

把图中顶点分成两组：

- **第一组**：已确定最短路径的顶点
- **第二组**：尚未确定最短路径的顶点

按最短路径长度递增顺序逐个把第二组的顶点加到第一组中，直至从$$s$$出发可以到达的所有顶点都包括进第一组。

在合并过程中，保持$$s$$到第一组各顶点的最短路径长度都不大于从$$s$$到第二组各顶点的最短路径长度。

- **第一组顶点对应的距离值**：从$$s$$到该顶点的最短路径长度
- **第二组顶点对应的距离值**：从$$s$$到该顶点的值包括第一组的顶点为中间顶点的最短路径长度

**具体过程：**

**初始化：** 第一组只包括源点$$s$$，第二组包括其它所有顶点。$$s$$距离值为0，第二组顶点的距离值确定如下：

- 若有边$$\langle s, V_i \rangle$$或$$(s, V_i)$$，则$$V_i$$的距离值为边所带的权，否则为$$\infty$$

**过程：** 每次从第二组的顶点中选一个其距离值为最小的顶点$$V_m$$加入到第一组中。

每往第一组加入顶点$$V_m$$，要对第二组各顶点的距离值进行一次修正：

- 若加进$$V_m$$做中间顶点，使从$$s$$到$$V_i$$的最短路径比不加$$V_m$$的短，则需要修改$$V_i$$的距离值

修改后再选距离值最小的顶点加入到第一组中，重复上述过程。

**结束条件：** 直到图的所有顶点都包括在第一组中或者再也没有可加入到第一组的顶点存在。

**最短路径的表示方法：**

借助一个长度为$$N$$的数组，包括：

1. 源点到当前节点的路径长度（`length`）
2. 当前节点的前驱节点（`pre`）

**Dist类定义：**

```cpp
class Dist {
public:
    int index;      // 顶点的索引值，仅Dijkstra算法用到
    int length;     // 当前最短路径长度
    int pre;        // 路径最后经过的顶点
};
```

**Dijkstra算法实现：**

```cpp
void Dijkstra(Graph& G, int s, Dist* &D) {
    D = new Dist[G.VerticesNum()];

    // 初始化Mark数组、D数组
    for (int i = 0; i < G.VerticesNum(); i++) {
        G.Mark[i] = UNVISITED;
        D[i].length = INFINITY;
        D[i].index = i;
        D[i].pre = s;
    }

    D[s].length = 0;
    MinHeap<Dist> H(G.EdgesNum());
    H.Insert(D[s]);

    for (int i = 0; i < G.VerticesNum(); i++) {
        bool FOUND = false;
        Dist d;

        while (!H.empty()) {
            d = H.RemoveMin();
            if (G.Mark[d.index] == UNVISITED) {
                FOUND = true;
                break;
            }
        }

        if (!FOUND) break;

        int v = d.index;
        G.Mark[v] = VISITED;
        Visit(v);

        // 更新权值等信息（松弛技术）
        for (Edge e = G.FirstEdge(v); G.IsEdge(e); e = G.NextEdge(e)) {
            if (D[G.ToVertex(e)].length > (D[v].length + G.Weight(e))) {
                D[G.ToVertex(e)].length = D[v].length + G.Weight(e);
                D[G.ToVertex(e)].pre = v;
                H.Insert(D[G.ToVertex(e)]);
            }
        }
    }
}
```

**时间复杂性分析：**

**不采用最小堆的方式：** 通过两两比较来扫描$$D$$数组。每次寻找权值最小结点，需要进行$$|V|$$次扫描，每次扫描$$|V|$$个顶点（$$|V|^2$$），而在更新$$D$$值处总共扫描$$|E|$$次。总时间代价为$$\Theta(|V|^2 + |E|) = \Theta(|V|^2)$$。

**采用最小堆的方式：** 每次改变$$D[i].\text{length}$$，通过先删除再重新插入的方法来改变顶点$$i$$在堆中的位置，或者仅为某个顶点添加一个新值（更小的），作为堆中新元素（而不作删除旧值的操作，因为旧值被找到时，该顶点一定被标记为`VISITED`，从而被忽略）。

不作删除旧值的缺点是，在最差情况下，它将使堆中元素数目由$$\Theta(|V|)$$增加到$$\Theta(|E|)$$，此时总的时间代价为$$\Theta((|V| + |E|)\log|E|)$$，因为处理每条边时都必须对堆进行一次重排。

### 7.5.3 Floyd算法

**求每对顶点间的最短路径：**

**方法1：** 反复执行Dijkstra算法

**方法2：** Floyd算法

**Floyd算法：**

该算法名称以创始人之一、1978年图灵奖获得者、斯坦福大学计算机科学系教授罗伯特·弗洛伊德命名。又称**插点法**，是一种用于寻找给定的加权图中任意节点对之间的最短路径算法。是一类**动态规划**的方法。

**算法过程：**

假设用邻接矩阵`adj`表示图。任意两点间距离是边的权，如果两点间没有边直接相连，则权为无穷大（$$\infty$$）。

在原图的邻接矩阵$$\text{adj}^{(0)}$$上做$$n$$次迭代，递归地产生一个矩阵序列$$\text{adj}^{(1)}, \text{adj}^{(2)}, \ldots, \text{adj}^{(n)}$$。

$$\text{adj}^{(k)}[i, j]$$等于从顶点$$V_i$$到顶点$$V_j$$中间顶点序号不大于$$k$$的最短路径长度。

$$\text{adj}^{(n)}$$包括了所有最终的最短路径。

**递推公式：**

假设已求得矩阵$$\text{adj}^{(k-1)}$$，那么从顶点$$V_i$$到顶点$$V_j$$中间顶点的序号不大于$$k$$的最短路径有两种情况：

1. **中间不经过顶点$$V_k$$**：那么就有$$\text{adj}^{(k)}[i, j] = \text{adj}^{(k-1)}[i, j]$$

2. **中间经过顶点$$V_k$$**：那么$$\text{adj}^{(k)}[i, j] < \text{adj}^{(k-1)}[i, j]$$，且$$\text{adj}^{(k)}[i, j] = \text{adj}^{(k-1)}[i, k] + \text{adj}^{(k-1)}[k, j]$$

**最短路径确定：**

设置一个$$n \times n$$的矩阵`path`，`path[i, j]`是由顶点$$V_i$$到顶点$$V_j$$的最短路径上排在顶点$$V_j$$前面的那个顶点，即当$$k$$是使得$$\text{adj}^{(k)}[i, j]$$达到最小值，那么就置`path[i, j] = path[k, j]`。

如果当前没有最短路径时，就将`path[i, j]`置为-1。

**Floyd算法实现：**

```cpp
void Floyd(Graph& G, Dist** &D) {
    int i, j, v;
    D = new Dist*[G.VerticesNum()];
    for (i = 0; i < G.VerticesNum(); i++)
        D[i] = new Dist[G.VerticesNum()];

    // 初始化D数组
    for (i = 0; i < G.VerticesNum(); i++)
        for (j = 0; j < G.VerticesNum(); j++)
            if (i == j) {
                D[i][j].length = 0;
                D[i][j].pre = i;
            } else {
                D[i][j].length = INFINITY;
                D[i][j].pre = -1;
            }

    // 矩阵初始化，仅初始化邻接顶点
    for (v = 0; v < G.VerticesNum(); v++)
        for (Edge e = G.FirstEdge(v); G.IsEdge(e); e = G.NextEdge(e)) {
            D[v][G.ToVertex(e)].length = G.Weight(e);
            D[v][G.ToVertex(e)].pre = v;
        }

    // 如果两顶点间最短路径经过顶点v，则有权值进行更新
    for (v = 0; v < G.VerticesNum(); v++)
        for (i = 0; i < G.VerticesNum(); i++)
            for (j = 0; j < G.VerticesNum(); j++)
                if ((D[i][v].length + D[v][j].length) < D[i][j].length) {
                    D[i][j].length = D[i][v].length + D[v][j].length;
                    D[i][j].pre = D[v][j].pre;
                }
}
```

**时间复杂性：**

三重`for`循环，复杂度是$$O(n^3)$$，适合稠密图。

## 7.6 最小支撑树

### 7.6.1 问题定义

**最小支撑树（Minimum-cost Spanning Tree, MST）：**

对于带权的**连通无向图**$$G$$，其最小支撑树是一个包括$$G$$的所有顶点和部分边的图，这部分的边满足下列条件：

- 保证图的连通性
- 边权值总和最小

**应用场景：**

公路网的造价问题、通信网络的设计等。

**代表算法：**

1. **Prim算法**
2. **Kruskal算法**

### 7.6.2 Prim算法

**具体操作：**

1. 从图中任意一个顶点开始，把这个顶点包括在MST里
2. 然后，在那些其一个端点已在MST里，另一个端点还未在MST里的边中，找权最小的一条边（相同边存在，任选择其一），并把这条边和其不在MST里的那个端点包括进MST里
3. 如此进行下去，每次往MST里加一个顶点和一条权最小的边，直到把所有的顶点都包括进MST里

**注意：** MST不唯一，但是最小权值是确定的。

**Prim算法实现：**

```cpp
void Prim(Graph& G, int s, Edge* &MST) {
    int MSTtag = 0;
    MST = new Edge[G.VerticesNum() - 1];
    MinHeap<Edge> H(G.EdgesNum());

    for (int i = 0; i < G.VerticesNum(); i++)
        G.Mark[i] = UNVISITED;

    int v = s;
    G.Mark[v] = VISITED;

    do {
        // 将以v为顶点，另一顶点未被标记的边插入最小值堆H
        for (Edge e = G.FirstEdge(v); G.IsEdge(e); e = G.NextEdge(e))
            if (G.Mark[G.ToVertex(e)] == UNVISITED)
                H.Insert(e);

        bool Found = false;
        Edge e;

        while (!H.empty()) {
            e = H.RemoveMin();
            if (G.Mark[G.ToVertex(e)] == UNVISITED) {
                Found = true;
                break;
            }
        }

        if (!Found) {
            Print("不存在最小支撑树。");
            delete[] MST;
            MST = NULL;
            return;
        }

        v = G.ToVertex(e);
        G.Mark[v] = VISITED;
        AddEdgetoMST(e, MST, MSTtag++);

    } while (MSTtag < (G.VerticesNum() - 1));
}
```

**证明：**

**引理：** 用Prim算法构造的生成树是MST。

首先证明这样一个结论：设$$T(V^*, E^*)$$是连通无向图$$G = (V, E)$$的一棵正在构造的生成树，又$$E$$中有边$$e = (V_x, V_y)$$，其中$$V_x \in V^*$$，$$V_y \notin V^*$$，且$$e$$的权$$W(e)$$是所有一个端点在$$V^*$$里，另一端不在$$V^*$$里的边的权中最小者，则一定存在$$G$$的一棵包括$$T$$的MST包括边$$e = (V_x, V_y)$$。

**反证法：**

1. 设$$G$$中任何包括$$T$$的MST都不包括$$e = (V_x, V_y)$$，且设$$T'$$是一棵这样的MST
2. 由于$$T'$$是连通的，因此有从$$V_x$$到$$V_y$$的路径$$V_x, \ldots, V_y$$
3. 把边$$e = (V_x, V_y)$$加进树$$T'$$，得到一个回路$$V_x, \ldots, V_y, V_x$$
4. 上述路径$$V_x, \ldots, V_y$$中必有边$$e' = (V_p, V_q)$$，其中$$V_p \in V^*$$，$$V_q \notin V^*$$，由条件知边的权$$W(e') \geq W(e)$$，从回路中去掉边$$e'$$，回路打开，成为另一棵生成树$$T''$$，$$T''$$包括边$$e = (V_x, V_y)$$，且各边权的总和不大于$$T'$$各边权总和
5. 因此$$T''$$是一棵包括边$$e$$的MST，与假设矛盾，即证明了我们的结论

**Prim算法与Dijkstra算法的区别：**

**相同点：** 都是贪心的思路

**不同点：**

- Prim算法要寻找的是离**已加入顶点**距离最近的点
- 而Dijkstra是寻找离**源点**距离最近的点

其时间复杂度分析与Dijkstra算法相同。

### 7.6.3 Kruskal算法

**基本思想：**

对于图$$G = (V, E)$$：

1. 开始时，将顶点集分为$$|V|$$个等价类，每个等价类包括一个顶点
2. 然后，以权的大小为顺序处理各条边，如果某条边连接两个不同等价类的顶点，则这条边被添加到MST，两个等价类被合并为一个
3. 反复执行此过程，直到只剩下一个等价类

**Kruskal算法实现：**

```cpp
void Kruskal(Graph& G, Edge* &MST) {
    Partree A(G.VerticesNum());                    // 等价类
    MinHeap<Edge> H(G.EdgesNum());                 // 声明一个最小堆
    MST = new Edge[G.VerticesNum() - 1];           // 最小支撑树
    int MSTtag = 0;                                // 最小支撑树边的标号

    // 将图的所有边插入最小值堆H中
    for (int i = 0; i < G.VerticesNum(); i++) {
        for (Edge e = G.FirstEdge(i); G.IsEdge(e); e = G.NextEdge(e))
            if (G.FromVertex(e) < G.ToVertex(e))
                H.Insert(e);
    }

    int EquNum = G.VerticesNum();                  // 开始时有|V|个等价类

    while (EquNum > 1) {
        Edge e = H.RemoveMin();                    // 获得下一条权最小的边
        int from = G.FromVertex(e);
        int to = G.ToVertex(e);

        if (A.differ(from, to)) {                  // 如果边e的两个顶点不在一个等价类
            A.UNION(from, to);                     // 将两个等价类合并为一个
            AddEdgetoMST(e, MST, MSTtag++);        // 将边e加到MST
            EquNum--;                              // 将等价类的个数减1
        }
    }
}
```

**性能分析：**

- 使用了路径压缩，`differ`和`UNION`函数几乎是常数
- 假设可能对几乎所有边都判断过了，则最坏情况下算法时间代价为$$O(|E|\log|E|)$$，即堆排序的时间
- **适合于稀疏图**

## 本章小结

本章介绍了图这一重要的非线性数据结构，主要内容包括：

1. **图的基本概念**：图的定义、分类、相关术语和性质
2. **图的存储结构**：邻接矩阵、邻接表和十字链表
3. **图的遍历**：深度优先搜索（DFS）和广度优先搜索（BFS）
4. **拓扑排序**：有向无环图的线性排序
5. **最短路径问题**：Dijkstra算法和Floyd算法
6. **最小支撑树**：Prim算法和Kruskal算法

**重要算法总结：**

| 算法        | 问题类型           | 算法思想      | 时间复杂度                                        | 适用场景           |
| ----------- | ------------------ | ------------- | ------------------------------------------------- | ------------------ |
| DFS         | 图遍历             | 递归深入      | $$O(\|V\| + \|E\|)$$                              | 连通性判断         |
| BFS         | 图遍历             | 队列逐层      | $$O(\|V\| + \|E\|)$$                              | 最短路径（无权图） |
| BFS-TopSort | 拓扑排序           | 入度为0       | $$O(\|V\| + \|E\|)$$                              | 任务调度           |
| DFS-TopSort | 拓扑排序           | 逆序输出      | $$O(\|V\| + \|E\|)$$                              | 任务调度           |
| Dijkstra    | 单源最短路径       | 贪心          | $$O(\|V\|^2)$$ 或 $$O((\|V\| + \|E\|)\log\|E\|)$$ | 非负权重图         |
| Floyd       | 所有顶点对最短路径 | 动态规划      | $$O(\|V\|^3)$$                                    | 稠密图             |
| Prim        | 最小支撑树         | 贪心          | $$O(\|V\|^2)$$ 或 $$O((\|V\| + \|E\|)\log\|E\|)$$ | 稠密图             |
| Kruskal     | 最小支撑树         | 贪心 + 并查集 | $$O(\|E\|\log\|E\|)$$                             | 稀疏图             |

**关键要点：**

- 图的存储结构选择取决于图的稠密程度：稠密图适合邻接矩阵，稀疏图适合邻接表
- 遍历算法是许多图算法的基础
- 贪心策略在最短路径和最小支撑树问题中的应用
- 动态规划在求解所有顶点对最短路径问题中的应用
- 并查集在Kruskal算法中的作用

图是计算机科学中最重要的数据结构之一，其应用遍及网络路由、社交网络分析、任务调度、推荐系统等众多领域。
