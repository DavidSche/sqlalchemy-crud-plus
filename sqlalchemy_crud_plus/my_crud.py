#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Generic, Iterable, Sequence, Type, Union

from sqlalchemy import Row, RowMapping, select, inspect
from sqlalchemy import delete as sa_delete
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy_crud_plus.errors import MultipleResultsError
from sqlalchemy_crud_plus.types import CreateSchema, Model, UpdateSchema
from sqlalchemy_crud_plus.my_utils import apply_sorting, count, parse_filters


class CRUDPlus(Generic[Model]):

    def __init__(self, model: Type[Model]):
        self.model = model
        self.primary_key = self._get_primary_key()

    def _get_primary_key(self):
        """
        Dynamically retrieve the primary key column(s) for the model.
        """
        mapper = inspect(self.model)
        primary_key = mapper.primary_key
        if len(primary_key) == 1:
            return primary_key[0]
        else:
            print(f'Composite primary keys are not supported,use {primary_key[0]}')
            return primary_key[0]
            # raise ValueError("Composite primary keys are not supported")

    async def _commit_if_needed(self, session: AsyncSession, commit: bool) -> None:
        """Helper method to commit a transaction if needed."""
        if commit:
            await session.commit()

    async def create_model(self, session: AsyncSession, obj: CreateSchema, commit: bool = False, **kwargs) -> Model:
        """
        Create a new instance of a model.
        """
        ins = self.model(**obj.model_dump(), **kwargs)
        session.add(ins)
        await self._commit_if_needed(session, commit)
        return ins

    async def create_models(self, session: AsyncSession, obj: Iterable[CreateSchema], commit: bool = False) -> list[Model]:
        """
        Create new instances of a model.
        """
        ins_list = [self.model(**ins.model_dump()) for ins in obj]
        session.add_all(ins_list)
        await self._commit_if_needed(session, commit)
        return ins_list

    async def select_model(self, session: AsyncSession, pk: Union[int, str]) -> Model | None:
        """
        Query by primary key.
        """
        stmt = select(self.model).where(self.primary_key == pk)
        query = await session.execute(stmt)
        return query.scalars().first()

    async def select_model_by_column(self, session: AsyncSession, **kwargs) -> Model | None:
        """Query a model instance by specific column(s)."""
        filters = await parse_filters(self.model, **kwargs)
        stmt = select(self.model).where(*filters)
        query = await session.execute(stmt)
        return query.scalars().first()

    async def select_models(self, session: AsyncSession, **kwargs) -> Sequence[Row[Any] | RowMapping | Any]:
        """Query multiple model instances based on filters."""
        filters = await parse_filters(self.model, **kwargs)
        stmt = select(self.model).where(*filters)
        query = await session.execute(stmt)
        return query.scalars().all()

    async def select_models_order(
        self, session: AsyncSession, sort_columns: str | list[str], sort_orders: str | list[str] | None = None, **kwargs
    ) -> Sequence[Row | RowMapping | Any] | None:
        """Query and sort multiple model instances based on filters."""
        filters = await parse_filters(self.model, **kwargs)
        stmt = select(self.model).where(*filters)
        stmt_sort = await apply_sorting(self.model, stmt, sort_columns, sort_orders)
        query = await session.execute(stmt_sort)
        return query.scalars().all()

    async def update_model(
        self, session: AsyncSession, pk: Union[int, str], obj: UpdateSchema | dict[str, Any], commit: bool = False
    ) -> int:
        """
        Update an instance by model's primary key.
        """
        instance_data = obj.model_dump(exclude_unset=True) if not isinstance(obj, dict) else obj
        stmt = sa_update(self.model).where(self.primary_key == pk).values(**instance_data)
        result = await session.execute(stmt)
        await self._commit_if_needed(session, commit)
        return result.rowcount # type: ignore

    async def update_model_by_column(
        self,
        session: AsyncSession,
        obj: UpdateSchema | dict[str, Any],
        allow_multiple: bool = False,
        commit: bool = False,
        **kwargs,
    ) -> int:
        """
        Update an instance by model column.
        """
        filters = await parse_filters(self.model, **kwargs)
        total_count = await count(session, self.model, filters)
        if not allow_multiple and total_count > 1:
            raise MultipleResultsError(f'Only one record is expected to be updated, found {total_count} records.')
        instance_data = obj.model_dump(exclude_unset=True) if not isinstance(obj, dict) else obj
        stmt = sa_update(self.model).where(*filters).values(**instance_data)
        result = await session.execute(stmt)
        await self._commit_if_needed(session, commit)
        return result.rowcount

    async def delete_model(self, session: AsyncSession, pk: Union[int, str], commit: bool = False) -> int:
        """
        Delete an instance by model's primary key.
        """
        stmt = sa_delete(self.model).where(self.primary_key == pk)
        result = await session.execute(stmt)
        await self._commit_if_needed(session, commit)
        return result.rowcount

    async def delete_model_by_column(
        self,
        session: AsyncSession,
        allow_multiple: bool = False,
        logical_deletion: bool = False,
        deleted_flag_column: str = 'del_flag',
        commit: bool = False,
        **kwargs,
    ) -> int:
        """
        Delete instances by model column, with optional logical deletion.
        """
        filters = await parse_filters(self.model, **kwargs)
        total_count = await count(session, self.model, filters)
        if not allow_multiple and total_count > 1:
            raise MultipleResultsError(f'Only one record is expected to be deleted, found {total_count} records.')
        if logical_deletion:
            deleted_flag = {deleted_flag_column: True}
            stmt = sa_update(self.model).where(*filters).values(**deleted_flag)
        else:
            stmt = sa_delete(self.model).where(*filters)
        result =await session.execute(stmt)
        await self._commit_if_needed(session, commit)
        return total_count
