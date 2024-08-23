#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
from typing import Any, Callable, Type, Union, List

from sqlalchemy import ColumnElement, Select, and_, asc, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.util import AliasedClass

from sqlalchemy_crud_plus.errors import ColumnSortError, ModelColumnError, SelectOperatorError
from sqlalchemy_crud_plus.types import Model

# Define supported SQLAlchemy operators
_SUPPORTED_FILTERS = {
    'gt': lambda col: col.__gt__,
    'lt': lambda col: col.__lt__,
    'ge': lambda col: col.__ge__,
    'le': lambda col: col.__le__,
    'eq': lambda col: col.__eq__,
    'ne': lambda col: col.__ne__,
    'between': lambda col: col.between,
    'in': lambda col: col.in_,
    'not_in': lambda col: col.not_in,
    'is': lambda col: col.is_,
    'is_not': lambda col: col.is_not,
    'is_distinct_from': lambda col: col.is_distinct_from,
    'is_not_distinct_from': lambda col: col.is_not_distinct_from,
    'like': lambda col: col.like,
    'not_like': lambda col: col.not_like,
    'ilike': lambda col: col.ilike,
    'not_ilike': lambda col: col.not_ilike,
    'startswith': lambda col: col.startswith,
    'endswith': lambda col: col.endswith,
    'contains': lambda col: col.contains,
    'match': lambda col: col.match,
    'concat': lambda col: col.concat,
    'add': lambda col: col.__add__,
    'radd': lambda col: col.__radd__,
    'sub': lambda col: col.__sub__,
    'rsub': lambda col: col.__rsub__,
    'mul': lambda col: col.__mul__,
    'rmul': lambda col: col.__rmul__,
    'truediv': lambda col: col.__truediv__,
    'rtruediv': lambda col: col.__rtruediv__,
    'floordiv': lambda col: col.__floordiv__,
    'rfloordiv': lambda col: col.__rfloordiv__,
    'mod': lambda col: col.__mod__,
    'rmod': lambda col: col.__rmod__,
}

async def get_sqlalchemy_filter(
    operator: str, value: Any, allow_arithmetic: bool = True
) -> Callable[[str], Callable] | None:
    """
    Retrieve the SQLAlchemy filter operator based on the provided operator string.
    """
    if operator in ['in', 'not_in', 'between'] and not isinstance(value, (tuple, list, set)):
        raise SelectOperatorError(f'The value for <{operator}> must be tuple, list, or set.')

    if operator in ['add', 'radd', 'sub', 'rsub', 'mul', 'rmul', 'truediv', 'rtruediv', 'floordiv', 'rfloordiv', 'mod', 'rmod'] and not allow_arithmetic:
        raise SelectOperatorError(f'Nested arithmetic operations are not allowed: {operator}')

    sqlalchemy_filter = _SUPPORTED_FILTERS.get(operator)
    if sqlalchemy_filter is None:
        warnings.warn(
            f'The operator <{operator}> is not supported. Supported operators: {", ".join(_SUPPORTED_FILTERS.keys())}.',
            SyntaxWarning,
        )
    return sqlalchemy_filter

async def get_column(model: Union[Type[Model], AliasedClass], field_name: str) -> ColumnElement:
    """
    Retrieve the column object for the specified field_name from the model.
    """
    column = getattr(model, field_name, None)
    if column is None:
        raise ModelColumnError(f'Column {field_name} not found in {model}.')
    return column

async def parse_filters(model: Union[Type[Model], AliasedClass], **kwargs) -> List[ColumnElement]:
    """
    Parse filter arguments into SQLAlchemy filter expressions.
    """
    filters = []
    for key, value in kwargs.items():
        if '__' in key:
            field_name, op = key.rsplit('__', 1)
            column = await get_column(model, field_name)
            sqlalchemy_filter = await get_sqlalchemy_filter(op, value)

            if op == 'or':
                or_filters = [
                    sqlalchemy_filter(column)(or_value)
                    for or_op, or_value in value.items()
                    if (sqlalchemy_filter := await get_sqlalchemy_filter(or_op, or_value)) is not None
                ]
                filters.append(or_(*or_filters))
            elif isinstance(value, dict) and {'value', 'condition'}.issubset(value):
                advanced_value = value['value']
                condition = value['condition']
                condition_filters = []
                for cond_op, cond_value in condition.items():
                    cond_filter = await get_sqlalchemy_filter(cond_op, cond_value, allow_arithmetic=False)
                    condition_filters.append(
                        cond_filter(sqlalchemy_filter(column)(advanced_value))(cond_value)
                        if cond_op != 'between'
                        else cond_filter(sqlalchemy_filter(column)(advanced_value))(*cond_value)
                    )
                filters.append(and_(*condition_filters))
            elif sqlalchemy_filter is not None:
                filters.append(
                    sqlalchemy_filter(column)(value) if op != 'between' else sqlalchemy_filter(column)(*value)
                )
        else:
            column = await get_column(model, key)
            filters.append(column == value)
    return filters

async def apply_sorting(
    model: Union[Type[Model], AliasedClass],
    stmt: Select,
    sort_columns: Union[str, List[str]],
    sort_orders: Union[str, List[str], None] = None,
) -> Select:
    """
    Apply sorting to a SQLAlchemy query based on specified column names and sort orders.
    """
    if sort_orders and not sort_columns:
        raise ValueError('Sort orders provided without corresponding sort columns.')

    if sort_columns:
        sort_columns = [sort_columns] if isinstance(sort_columns, str) else sort_columns
        validated_sort_orders = ['asc'] * len(sort_columns)

        if sort_orders:
            sort_orders = [sort_orders] if isinstance(sort_orders, str) else sort_orders
            if len(sort_columns) != len(sort_orders):
                raise ColumnSortError('The length of sort_columns and sort_orders must match.')
            validated_sort_orders = sort_orders

        for column_name, order in zip(sort_columns, validated_sort_orders):
            column = await get_column(model, column_name)
            stmt = stmt.order_by(asc(column) if order == 'asc' else desc(column))
    return stmt

async def count(
    session: AsyncSession,
    model: Union[Type[Model], AliasedClass],
    filters: List[ColumnElement],
) -> int:
    """
    Count records that match specified filters.
    :param session: The sqlalchemy session to use for the operation.
    :param model: The SQLAlchemy model.
    :param filters: Filters to apply for the count.
    :return:
    """
    stmt = select(func.count()).select_from(model)
    if filters:
        stmt = stmt.where(*filters)
    query = await session.execute(stmt)
    return query.scalar() or 0

# Key Optimizations:
# Type Annotations: Updated type annotations for better clarity and to support different types in Union.
# Error Handling: Improved error handling, especially when dealing with sort_columns and sort_orders.
# Simplification: Simplified complex logic, particularly in parse_filters and apply_sorting, for easier understanding and maintenance.
# Consistency: Ensured consistent use of Union, List, and Callable for better readability and to follow modern Python conventions.
# Documentation: Added or updated docstrings to describe the purpose and behavior of each function clearly.