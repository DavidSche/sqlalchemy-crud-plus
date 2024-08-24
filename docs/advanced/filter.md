SQLAlchemy CRUD Plus 支持高级过滤选项，允许使用运算符查询记录，如大于（`__gt`）、小于（`__lt`）；

大多数过滤器操作符需要一个字符串或整数值

```python
# 获取年龄大于 30 岁以上的员工
items = await item_crud.select_models(
    session=db,
    age__gt=30,
)
```

## 比较运算符

- `__gt`：大于
- `__lt`：小于
- `__ge`：大于或等于
- `__le`：小于或等于
- `__eq`: 等于
- `__ne`: 不等于
- `__between`: 在两者之间

## IN 比较

- `__in`: 包含
- `__not_in`: 不包括

## 身份比较

- `__is`：用于测试 “真”、“假” 和 “无”。
- `__is_not`：“is” 的否定
- `__is_distinct_from`: 产生 SQL IS DISTINCT FROM
- `__is_not_distinct_from`: Produces SQL IS NOT DISTINCT FROM
- `__like`：针对特定文本模式的 SQL “like” 搜索
- `__not_like`：“like” 的否定
- `__ilike`：大小写不敏感的 “like”
- `__not_ilike`：大小写不敏感的 “not_like”

## 字符串比较

- `__startswith`：文本以给定的字符串开始
- `__endswith`：文本以给定字符串结束
- `__contains`：文本包含给定字符串

## 字符串匹配

- `__match`：特定于数据库的匹配表达式

## 字符串修改

- `__concat`: 字符串连接

## 算术运算符

此过滤器使用方法需查看：[算数](#_8)

- `__add`: Python `+` 运算符
- `__radd`: Python `+` 反向运算
- `__sub`: Python `-` 运算符
- `__rsub`: Python `-` 反向运算
- `__mul`: Python `*` 运算符
- `__rmul`: Python `*` 反向运算
- `__truediv`: Python `/` 运算符，这是 Python 的 truediv 操作符，它将确保发生整数真除法
- `__rtruediv`: Python `/` 反向运算
- `__floordiv`: Python `//` operator，这是 Python 的 floordiv 运算符，它将确保发生底除
- `__rfloordiv`: Python `//` 反向运算
- `__mod`: Python `%` 运算符
- `__rmod`: Python `%` 反向运算

## BETWEEN、IN、NOT IN

!!! note

    运算符需要多个值，且仅允许元组，列表，集合

```python
# 获取年龄在 30 - 40 岁之间的员工
items = await item_crud.select_models(
    session=db,
    age__between=[30, 40],
)
```

## AND

可以通过将多个过滤器链接在一起来实现 AND 子句

```python
# 获取年龄在 30 以上，薪资大于 2w 的员工
items = await item_crud.select_models(
    session=db,
    age__gt=30,
    payroll__gt=20000,
)
```

## OR

!!! note

    每个键都应是库已支持的过滤器，仅允许字典

```python
# 获取年龄在 40 岁以上或 30 岁以下的员工
items = await item_crud.select_models(
    session=db,
    age__or={'gt': 40, 'lt': 30},
)
```

## 算数

!!! note

    此过滤器必须传递字典，且字典结构必须为 `{'value': xxx, 'condition': {'已支持的过滤器': xxx}}`

    `value`：此值将与列值进行运算

    `condition`：此值将作为运算后的比较值，比较条件取决于使用的过滤器

```python
# 获取薪资打八折以后仍高于 15000 的员工
items = await item_crud.select_models(
    session=db,
    payroll__mul={'value': 0.8, 'condition': {'gt': 15000}},
)
```