"""LLM tools for Grocery Intel."""
from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.components import llm
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.llm import LLMContext

from .const import DOMAIN
from .read_model import (
    ANALYTICS_SCOPE,
    DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT,
    SCOPES,
    aggregate_public_read_model,
    async_build_public_read_model,
    describe_public_schema,
    normalize_scope,
    query_public_read_model,
)


PROMPT = (
    "Grocery Intel exposes a read-only, versioned public data model for household "
    "receipt, grocery, dining, product, store, price, and inventory analysis. "
    "Use GroceryIntelDescribeSchema before unfamiliar queries. Prefer "
    "GroceryIntelQuery for row-level lookup and GroceryIntelAggregate for grouped "
    "analytics. Treat results as evidence-limited: mention sample sizes, matched "
    "rows, and data coverage when making comparisons."
)


def _scope_schema() -> Any:
    return vol.In(sorted(SCOPES))


def _get_data(hass: HomeAssistant) -> Any:
    domain_data = hass.data.get(DOMAIN)
    if not isinstance(domain_data, dict) or not domain_data:
        raise HomeAssistantError("Grocery Intel is not loaded")
    data = next(iter(domain_data.values()))
    if data is None:
        raise HomeAssistantError("Grocery Intel data is not available")
    return data


async def _async_read_model(hass: HomeAssistant, scope: str) -> dict[str, Any]:
    data = _get_data(hass)
    return await async_build_public_read_model(data, scope=scope)


class GroceryIntelDescribeSchemaTool(llm.Tool):
    """Describe the Grocery Intel public read model."""

    name = "GroceryIntelDescribeSchema"
    description = (
        "Describe the read-only Grocery Intel public schema, datasets, fields, "
        "relationships, filters, aggregation options, and privacy scopes."
    )
    parameters = vol.Schema(
        {
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: LLMContext,
    ) -> dict[str, Any]:
        """Return the schema descriptor."""
        scope = normalize_scope(tool_input.tool_args.get("scope"))
        return describe_public_schema(scope=scope)


class GroceryIntelQueryTool(llm.Tool):
    """Query the Grocery Intel public read model."""

    name = "GroceryIntelQuery"
    description = (
        "Query rows from a Grocery Intel dataset. Supports filters, sorting, "
        "pagination, field selection, and related record inclusion. This is "
        "read-only and uses the public privacy-filtered data model."
    )
    parameters = vol.Schema(
        {
            vol.Required("dataset"): str,
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("filters"): object,
            vol.Optional("sort"): object,
            vol.Optional("limit", default=DEFAULT_QUERY_LIMIT): vol.Coerce(int),
            vol.Optional("offset", default=0): vol.Coerce(int),
            vol.Optional("fields"): object,
            vol.Optional("include"): object,
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: LLMContext,
    ) -> dict[str, Any]:
        """Return matching rows from a public dataset."""
        args = tool_input.tool_args
        scope = normalize_scope(args.get("scope"))
        read_model = await _async_read_model(hass, scope)
        try:
            return query_public_read_model(
                read_model,
                dataset=str(args.get("dataset") or ""),
                filters=args.get("filters"),
                sort=args.get("sort"),
                limit=args.get("limit", DEFAULT_QUERY_LIMIT),
                offset=args.get("offset", 0),
                fields=args.get("fields"),
                include=args.get("include"),
            )
        except ValueError as err:
            raise HomeAssistantError(str(err)) from err


class GroceryIntelAggregateTool(llm.Tool):
    """Aggregate the Grocery Intel public read model."""

    name = "GroceryIntelAggregate"
    description = (
        "Aggregate a Grocery Intel dataset with filters, group-by fields, "
        "metrics, optional time buckets, sorting, and row limits. Use this for "
        "open-ended analytics such as spend by store, product price movement, "
        "or month-over-month trends."
    )
    parameters = vol.Schema(
        {
            vol.Required("dataset"): str,
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("filters"): object,
            vol.Optional("group_by"): object,
            vol.Optional("metrics"): object,
            vol.Optional("time_bucket"): object,
            vol.Optional("sort"): object,
            vol.Optional("limit", default=DEFAULT_QUERY_LIMIT): vol.Coerce(int),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: LLMContext,
    ) -> dict[str, Any]:
        """Return grouped analytics from a public dataset."""
        args = tool_input.tool_args
        scope = normalize_scope(args.get("scope"))
        read_model = await _async_read_model(hass, scope)
        try:
            return aggregate_public_read_model(
                read_model,
                dataset=str(args.get("dataset") or ""),
                filters=args.get("filters"),
                group_by=args.get("group_by"),
                metrics=args.get("metrics"),
                time_bucket=args.get("time_bucket"),
                sort=args.get("sort"),
                limit=args.get("limit", DEFAULT_QUERY_LIMIT),
            )
        except ValueError as err:
            raise HomeAssistantError(str(err)) from err


class GroceryIntelExportReadModelTool(llm.Tool):
    """Return a capped public read-model export."""

    name = "GroceryIntelExportReadModel"
    description = (
        "Return a capped read-only export of the Grocery Intel public read "
        "model. Use for consumers that need broad local context, but prefer "
        "GroceryIntelQuery or GroceryIntelAggregate for targeted analysis."
    )
    parameters = vol.Schema(
        {
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("datasets"): object,
            vol.Optional("limit_per_dataset", default=DEFAULT_QUERY_LIMIT): vol.Coerce(int),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: LLMContext,
    ) -> dict[str, Any]:
        """Return capped read-model datasets."""
        args = tool_input.tool_args
        scope = normalize_scope(args.get("scope"))
        read_model = await _async_read_model(hass, scope)
        requested = _coerce_string_set(args.get("datasets"))
        limit = max(1, min(int(args.get("limit_per_dataset", DEFAULT_QUERY_LIMIT)), MAX_QUERY_LIMIT))
        datasets = read_model.get("datasets") or {}
        selected: dict[str, list[dict[str, Any]]] = {}
        for name, rows in datasets.items():
            if requested and name not in requested:
                continue
            selected[name] = list(rows or [])[:limit]
        return {
            "schema_version": read_model.get("schema_version"),
            "scope": read_model.get("scope"),
            "counts": read_model.get("counts"),
            "limit_per_dataset": limit,
            "datasets": selected,
        }


def _coerce_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value.strip() else set()
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


@callback
def async_get_tools(hass: HomeAssistant, llm_context: LLMContext) -> llm.LLMTools:
    """Return Grocery Intel tools to expose to Home Assistant LLM APIs."""
    return llm.LLMTools(
        tools=[
            GroceryIntelDescribeSchemaTool(),
            GroceryIntelQueryTool(),
            GroceryIntelAggregateTool(),
            GroceryIntelExportReadModelTool(),
        ],
        prompt=PROMPT,
    )
