# TensorLanguage (TL) 中的逻辑编程

TL 将强大的 Prolog 风格逻辑推理引擎直接集成到其运行时（张量计算引擎）中。这使得您可以定义知识库、执行逻辑推理，并将推理结果（符号推理）与神经网络和数值计算无缝结合。

## 1. 语法概述

在 TL 中，逻辑语句被视为一等公民。您可以定义**事实 (Facts)**、**规则 (Rules)** 和**查询 (Queries)**。

### 事实 (Facts)
事实声明静态知识。它们由谓词和参数（实体或值）组成。

```rust
// 语法糖（推荐）
father(alice, bob).       // "alice 是 bob 的父亲"
is_student(charlie).      // 一元谓词
```

### 关系声明 (Relation Declarations)
可以显式声明关系的参数类型。如果未声明，则从规则和事实中自动推断。

```rust
relation parent(entity, entity);
relation age(entity, i64);
```

### 规则 (Rules)
规则定义如何从现有事实推导新事实。如果**体**（右侧）为真，则推断**头**（左侧）也为真。

```rust
// "如果 x 是 y 的父亲，且 y 是 z 的父亲，则 x 是 z 的祖父"
grandparent(x, z) :- father(x, y), father(y, z).

// 也支持递归规则
ancestor(x, y) :- father(x, y).
ancestor(x, y) :- father(x, z), ancestor(z, y).
```

- 规则中的变量以小写字母开头（惯例为 `x`、`y`、`z`）。
- `,` 表示逻辑合取（AND）。
- `.` 终止语句。

### 查询 (Queries)
使用 `?` 后缀查询知识库 (KB)。查询结果以张量形式返回。

```rust
// 1. 真/假查询
// 返回 0 维张量：[1.]（真）或 [0.]（假）
let is_father = ?father(alice, bob); 
println("Is alice father of bob? {}", is_father);

// 2. 变量查询（搜索）
// 使用 $变量名 来询问"谁？"或"什么？"。返回匹配列表。
// 结果：形状为 [N, 1] 的张量。包含实体 ID（显示为名称）。
let children = ?father(alice, $child);
println("Children of alice: {}", children);
```

## 2. 符号输出

TL 自动将实体名称（符号）内部映射为唯一的整数 ID。显示逻辑查询结果（包含实体 ID 的张量）时，运行时会自动将这些 ID 解析回原始名称。

```rust
father(alice, bob).

fn main() {
    println("{}", ?father(alice, $x));
    // 输出：
    // [[bob]]
}
```

## 3. 作用域和文件组织

事实和规则必须在**全局作用域**（函数外部）定义。不能在函数内部定义。

### 单文件（脚本风格）
事实、规则和 `main` 函数可以全部写在同一个文件中。

```rust
// main.tl
father(alice, bob).
grandparent(x, z) :- father(x, y), father(y, z).

fn main() {
    let res = ?grandparent(alice, $x);
    println("{}", res);
}
```

### 外部文件（模块风格）
逻辑可以组织到单独的文件中。编译器会自动从所有导入的模块中收集事实和规则。

**facts.tl**：
```rust
father(alice, bob).
father(bob, charlie).
```

**logic.tl**：
```rust
// 规则也可以放在单独的文件中
grandparent(x, z) :- father(x, y), father(y, z).
```

**main.tl**：
```rust
mod facts;
mod logic;

// 将关系和规则导入当前作用域
use facts::*;
use logic::*;

fn main() {
    // 'facts.tl' 的事实和 'logic.tl' 的规则会自动加载
    let res = ?grandparent(alice, $x);
    println("{}", res);
}
```

## 4. 与张量的集成

查询结果是标准的 TL 张量，因此可以直接用于数学和神经网络运算。

- **布尔值**：`0.0` 或 `1.0`（浮点数）。适用于掩码和条件逻辑。
- **搜索结果**：实体 ID 的 `Int64` 张量。可用作嵌入层的索引。

示例：神经符号集成
```rust
// 逻辑：搜索所有祖先
let ancestors = ?ancestor(alice, $x);

// 神经：获取这些祖先的嵌入向量
let embeds = embedding(ancestors, weights);

// 计算平均向量
let query_vec = mean(embeds, 0);
```

## 4. 高级功能

### 传递和递归逻辑
TL 支持完整的 Datalog 风格递归。您可以定义复杂的关系，如图的可达性和传递闭包。

```rust
path(x, y) :- edge(x, y).
path(x, y) :- edge(x, z), path(z, y).
```

### 多重依赖
规则可以有多个条件。只有当所有条件都满足时，规则才成立。

```rust
compatible(x, y) :- is_friend(x, y), has_same_hobby(x, y).
```

### 自动知识库初始化
无需手动初始化知识库。编译器会自动从所有模块（包括导入的模块）中聚合事实和规则，并在 `main()` 开始时初始化推理引擎。

## 5. 否定和算术比较

### 否定 (Negation)
可以在规则体中使用 `not()` 来表达事实的否定。TL 支持分层否定（Stratified Negation）。

```rust
// 没有直属上司的人是管理者
manager(X) :- employee(X), not(has_boss(X)).
```

否定可以与递归结合使用，但否定循环（negative cycle）会被检测到并导致编译错误。

```rust
// 这是有效的（否定在不同的层中）
reachable(X, Y) :- edge(X, Y).
reachable(X, Y) :- edge(X, Z), reachable(Z, Y).
unreachable(X, Y) :- node(X), node(Y), not(reachable(X, Y)).
```

### 算术比较
可以在规则体中使用算术比较作为条件。

```rust
// 18 岁及以上为成人
adult(X) :- person(X, Age), Age >= 18.

// 计算薪资差异
earns_more(X, Y) :- salary(X, Sx), salary(Y, Sy), Sx > Sy.
```

支持的比较运算符：`>`、`<`、`>=`、`<=`、`==`、`!=`
