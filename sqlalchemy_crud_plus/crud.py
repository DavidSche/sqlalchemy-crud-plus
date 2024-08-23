#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Generic, Iterable, Sequence, Type, Union

from sqlalchemy import Row, RowMapping, select, update as sa_update, delete as sa_delete, inspect
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy_crud_plus.errors import MultipleResultsError
from sqlalchemy_crud_plus.types import CreateSchema, Model, UpdateSchema
from sqlalchemy_crud_plus.utils import apply_sorting, count, parse_filters


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
        """Create a new instance of a model."""
        data = {**obj.model_dump(), **kwargs}
        ins = self.model(**data)
        session.add(ins)
        await self._commit_if_needed(session, commit)
        return ins

    async def create_models(self, session: AsyncSession, objs: Iterable[CreateSchema], commit: bool = False) -> list[
        Model]:
        """Create multiple instances of a model."""
        instances = [self.model(**obj.model_dump()) for obj in objs]
        session.add_all(instances)
        await self._commit_if_needed(session, commit)
        return instances

    async def select_model(self, session: AsyncSession, pk: Union[int, str]) -> Union[Model, None]:
        """Query a model instance by its primary key."""
        stmt = select(self.model).where(self.primary_key == pk)
        result = await session.execute(stmt)
        return result.scalars().first()

    async def select_model_by_column(self, session: AsyncSession, **kwargs) -> Union[Model, None]:
        """Query a model instance by specific column(s)."""
        filters = await parse_filters(self.model, **kwargs)
        stmt = select(self.model).where(*filters)
        result = await session.execute(stmt)
        return result.scalars().first()

    async def select_models(self, session: AsyncSession, **kwargs) -> Sequence[Union[Row[Any], RowMapping, Any]]:
        """Query multiple model instances based on filters."""
        filters = await parse_filters(self.model, **kwargs)
        stmt = select(self.model).where(*filters)
        result = await session.execute(stmt)
        return result.scalars().all()

    async def select_models_order(
            self, session: AsyncSession, sort_columns: Union[str, list[str]],
            sort_orders: Union[str, list[str], None] = None, **kwargs
    ) -> Sequence[Union[Row, RowMapping, Any]]:
        """Query and sort multiple model instances based on filters."""
        filters = await parse_filters(self.model, **kwargs)
        stmt = select(self.model).where(*filters)
        stmt = await apply_sorting(self.model, stmt, sort_columns, sort_orders)
        result = await session.execute(stmt)
        return result.scalars().all()

    async def update_model(
            self, session: AsyncSession, pk: Union[int, str], obj: Union[UpdateSchema, dict[str, Any]], commit: bool = False
    ) -> int:
        """Update a model instance by its primary key."""
        instance_data = obj.model_dump(exclude_unset=True) if not isinstance(obj, dict) else obj
        stmt = sa_update(self.model).where(self.primary_key == pk).values(**instance_data)
        result = await session.execute(stmt)
        await self._commit_if_needed(session, commit)
        return result.rowcount  # type: ignore

    async def update_model_by_column(
            self, session: AsyncSession, obj: Union[UpdateSchema, dict[str, Any]], allow_multiple: bool = False,
            commit: bool = False, **kwargs
    ) -> int:
        """Update model instances by specific column(s)."""
        filters = await parse_filters(self.model, **kwargs)
        total_count = await count(session, self.model, filters)
        if not allow_multiple and total_count > 1:
            raise MultipleResultsError(f'Only one record is expected to be updated, found {total_count} records.')

        instance_data = obj.model_dump(exclude_unset=True) if not isinstance(obj, dict) else obj
        stmt = sa_update(self.model).where(*filters).values(**instance_data)
        result = await session.execute(stmt)
        await self._commit_if_needed(session, commit)
        return result.rowcount  # type: ignore

    async def delete_model(self, session: AsyncSession, pk: Union[int, str], commit: bool = False) -> int:
        """Delete a model instance by its primary key."""
        stmt = sa_delete(self.model).where(self.primary_key == pk)
        result = await session.execute(stmt)
        await self._commit_if_needed(session, commit)
        return result.rowcount  # type: ignore

    async def delete_model_by_column(
            self, session: AsyncSession, allow_multiple: bool = False, logical_deletion: bool = False,
            deleted_flag_column: str = 'del_flag', commit: bool = False, **kwargs
    ) -> int:
        """Delete model instances by specific column(s), supports logical deletion."""
        filters = await parse_filters(self.model, **kwargs)
        total_count = await count(session, self.model, filters)
        if not allow_multiple and total_count > 1:
            raise MultipleResultsError(f'Only one record is expected to be deleted, found {total_count} records.')

        if logical_deletion:
            stmt = sa_update(self.model).where(*filters).values(**{deleted_flag_column: True})
        else:
            stmt = sa_delete(self.model).where(*filters)

        result = await session.execute(stmt)
        await self._commit_if_needed(session, commit)
        return total_count
