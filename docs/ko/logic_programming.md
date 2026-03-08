# TensorLanguage (TL)에서의 논리 프로그래밍

TL은 강력한 Prolog 스타일의 논리 추론 엔진을 런타임(텐서 계산 엔진)에 직접 통합하고 있습니다. 이를 통해 지식 베이스를 정의하고, 논리적 추론을 수행하며, 그 결과(기호 추론)를 신경망 및 수치 계산과 원활하게 결합할 수 있습니다.

## 1. 구문 개요

TL에서 논리 문은 일급 시민으로 취급됩니다. **사실 (Facts)**, **규칙 (Rules)**, **쿼리 (Queries)**를 정의할 수 있습니다.

### 사실 (Facts)
사실은 정적 지식을 선언합니다. 술어와 인자(엔티티 또는 값)로 구성됩니다.

```rust
father(alice, bob).       // "alice는 bob의 아버지이다"
is_student(charlie).      // 단항 술어
```

### 관계 선언 (Relation Declarations)
관계의 인자 유형을 명시적으로 선언할 수 있습니다. 선언하지 않으면 규칙과 사실에서 자동 추론됩니다.

```rust
relation parent(entity, entity);
relation age(entity, i64);
```

### 규칙 (Rules)
규칙은 기존 사실에서 새로운 사실을 도출하는 방법을 정의합니다.

```rust
grandparent(x, z) :- father(x, y), father(y, z).
ancestor(x, y) :- father(x, y).
ancestor(x, y) :- father(x, z), ancestor(z, y).
```

### 쿼리 (Queries)
`?` 접미사를 사용하여 지식 베이스(KB)를 쿼리합니다. 결과는 텐서로 반환됩니다.

```rust
let is_father = ?father(alice, bob); 
let children = ?father(alice, $child);
```

## 2. 기호 출력

TL은 엔티티 이름(기호)을 내부적으로 고유한 정수 ID에 자동으로 매핑합니다. 표시 시 런타임이 자동으로 원래 이름으로 해석합니다.

## 3. 범위와 파일 구성

사실과 규칙은 **전역 범위**(함수 외부)에서 정의해야 합니다.

### 모듈 스타일
```rust
mod facts;
mod logic;
use facts::*;
use logic::*;
```

## 4. 텐서와의 통합

쿼리 결과는 표준 TL 텐서이므로 수학적 및 신경망 연산에 직접 사용할 수 있습니다.

## 5. 부정과 산술 비교

### 부정 (Negation)
규칙 본체에서 `not()`를 사용하여 사실의 부정을 표현할 수 있습니다. TL은 계층화 부정(Stratified Negation)을 지원합니다.

```rust
manager(X) :- employee(X), not(has_boss(X)).
```

### 산술 비교
```rust
adult(X) :- person(X, Age), Age >= 18.
```

지원되는 비교 연산자: `>`, `<`, `>=`, `<=`, `==`, `!=`
