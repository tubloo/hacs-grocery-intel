"""LLM tools for Grocery Intel."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import voluptuous as vol

from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from .const import DOMAIN
from .read_model import (
    ANALYTICS_SCOPE,
    DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT,
    SCOPES,
    aggregate_public_read_model,
    async_build_public_read_model,
    calculate_grocery_spend_summary,
    describe_public_schema,
    find_product_price_history,
    get_grocery_spend_breakdown,
    inspect_grocery_data_quality,
    list_recent_grocery_receipts,
    normalize_scope,
    query_public_read_model,
)


def _api_prompt(hass: HomeAssistant, llm_context: llm.LLMContext | None = None) -> str:
    """Build language-aware instructions for Grocery Intel tools."""
    requested_language = (llm_context.language if llm_context else None) or ""
    ha_language = getattr(hass.config, "language", None) or ""
    return (
        "Grocery Intel exposes a read-only, versioned public data model for household "
        "receipt, grocery, dining, product, store, price, and inventory analysis. "
        "Use DescribeGroceryIntelDataModel before unfamiliar queries. Prefer "
        "CalculateGrocerySpendSummary for simple spend totals, "
        "ListRecentGroceryReceipts for latest purchases, GetGrocerySpendBreakdown "
        "for store/category/month summaries, and FindProductPriceHistory for "
        "product price questions. Prefer CalculateGroceryIntelAnalytics only for "
        "advanced custom aggregations. Prefer SearchGroceryIntelData for row-level "
        "lookup, specific products, or source details. "
        "Use ExportGroceryIntelDataSnapshot only when targeted query or aggregate "
        "calls are insufficient. Treat results as evidence-limited: mention date "
        "ranges, filters, sample sizes, matched rows, and data coverage when making "
        "comparisons. "
        f"Requested response language is '{requested_language or 'not specified'}'. "
        f"Home Assistant language is '{ha_language or 'not specified'}'. Use the "
        "requested response language when provided; otherwise follow the user's "
        "conversation language. Treat Home Assistant language as locale context only, "
        "not as a hard requirement for answer language. Keep tool names, dataset "
        "names, field names, JSON keys, store names, product names, and raw data "
        "values unchanged."
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


LLM_API_ID = f"{DOMAIN}.public_read_model"
LLM_API_NAME = "Grocery Intel"


class GroceryIntelDescribeSchemaTool(llm.Tool):
    """Describe the Grocery Intel public read model."""

    name = "DescribeGroceryIntelDataModel"
    description = (
        "Describe the Grocery Intel read-only data model and query options. "
        "Use this first when you need dataset names, field names, relationships, "
        "filter operators, aggregation metrics, time bucket options, or privacy "
        "scopes before searching or calculating grocery analytics."
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
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Return the schema descriptor."""
        scope = normalize_scope(tool_input.tool_args.get("scope"))
        return describe_public_schema(scope=scope)


class GroceryIntelQueryTool(llm.Tool):
    """Query the Grocery Intel public read model."""

    name = "SearchGroceryIntelData"
    description = (
        "Search Grocery Intel receipts, receipt line items, products, purchase "
        "observations, stores, inventory images, and full-scope activities. "
        "Use this for row-level questions such as recent receipts, receipts from "
        "a date range or store, specific product history, source rows supporting "
        "an answer, or selected fields from a dataset. Supports filters, sorting, "
        "pagination, field selection, and related record inclusion."
    )
    parameters = vol.Schema(
        {
            vol.Required("dataset"): str,
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("filters"): vol.Any(dict, list),
            vol.Optional("sort"): vol.Any(dict, list, str),
            vol.Optional("limit", default=DEFAULT_QUERY_LIMIT): vol.Coerce(int),
            vol.Optional("offset", default=0): vol.Coerce(int),
            vol.Optional("fields"): vol.Any(list, str),
            vol.Optional("include"): vol.Any(list, str),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
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

    name = "CalculateGroceryIntelAnalytics"
    description = (
        "Calculate Grocery Intel analytics from receipts, products, stores, and "
        "purchase observations. Use this for spend totals, spend by store, "
        "category summaries, week/month/year trends, product price comparisons, "
        "cheapest-store analysis, and price changes over time. Supports filters, "
        "group-by fields, count/sum/avg/min/max metrics, day/week/month/year time "
        "buckets, sorting, and row limits."
    )
    parameters = vol.Schema(
        {
            vol.Required("dataset"): str,
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("filters"): vol.Any(dict, list),
            vol.Optional("group_by"): vol.Any(list, str),
            vol.Optional("metrics"): vol.Any(dict, list, str),
            vol.Optional("time_bucket"): dict,
            vol.Optional("sort"): vol.Any(dict, list, str),
            vol.Optional("limit", default=DEFAULT_QUERY_LIMIT): vol.Coerce(int),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
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


class GroceryIntelSpendSummaryTool(llm.Tool):
    """Calculate common grocery spend summaries with scalar parameters."""

    name = "CalculateGrocerySpendSummary"
    description = (
        "Calculate total grocery spend and receipt count for a date range using "
        "simple scalar parameters. Prefer this for questions like this week, this "
        "month, last month, this year, spend at a store, or spend by category."
    )
    parameters = vol.Schema(
        {
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("start_date"): str,
            vol.Optional("end_date"): str,
            vol.Optional("date_field", default="purchased_at"): vol.In(["purchased_at", "created_at"]),
            vol.Optional("store_name"): str,
            vol.Optional("category"): str,
            vol.Optional("subcategory"): str,
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Return a spend summary."""
        args = tool_input.tool_args
        scope = normalize_scope(args.get("scope"))
        read_model = await _async_read_model(hass, scope)
        return calculate_grocery_spend_summary(
            read_model,
            start_date=args.get("start_date"),
            end_date=args.get("end_date"),
            date_field=str(args.get("date_field") or "purchased_at"),
            store_name=args.get("store_name"),
            category=args.get("category"),
            subcategory=args.get("subcategory"),
        )


class GroceryIntelRecentReceiptsTool(llm.Tool):
    """List recent grocery receipts with scalar parameters."""

    name = "ListRecentGroceryReceipts"
    description = (
        "List recent or previous grocery receipts using simple scalar parameters. "
        "Use this for last spend, spend before a known receipt, recent purchases, "
        "or checking date coverage."
    )
    parameters = vol.Schema(
        {
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("limit", default=10): vol.Coerce(int),
            vol.Optional("start_date"): str,
            vol.Optional("end_date"): str,
            vol.Optional("before_date"): str,
            vol.Optional("date_field", default="purchased_at"): vol.In(["purchased_at", "created_at"]),
            vol.Optional("store_name"): str,
            vol.Optional("category"): str,
            vol.Optional("include_missing_dates", default=False): bool,
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Return recent receipts."""
        args = tool_input.tool_args
        scope = normalize_scope(args.get("scope"))
        read_model = await _async_read_model(hass, scope)
        return list_recent_grocery_receipts(
            read_model,
            limit=args.get("limit", 10),
            start_date=args.get("start_date"),
            end_date=args.get("end_date"),
            before_date=args.get("before_date"),
            date_field=str(args.get("date_field") or "purchased_at"),
            store_name=args.get("store_name"),
            category=args.get("category"),
            include_missing_dates=bool(args.get("include_missing_dates", False)),
        )


class GroceryIntelSpendBreakdownTool(llm.Tool):
    """Calculate grocery spend breakdowns with scalar grouping."""

    name = "GetGrocerySpendBreakdown"
    description = (
        "Calculate spend grouped by one common dimension using scalar parameters. "
        "Use this for spend by store, category, subcategory, week, month, or year."
    )
    parameters = vol.Schema(
        {
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("start_date"): str,
            vol.Optional("end_date"): str,
            vol.Optional("date_field", default="purchased_at"): vol.In(["purchased_at", "created_at"]),
            vol.Optional("group_by", default="store"): vol.In(
                ["store", "category", "subcategory", "week", "month", "year"]
            ),
            vol.Optional("store_name"): str,
            vol.Optional("category"): str,
            vol.Optional("limit", default=100): vol.Coerce(int),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Return a spend breakdown."""
        args = tool_input.tool_args
        scope = normalize_scope(args.get("scope"))
        read_model = await _async_read_model(hass, scope)
        return get_grocery_spend_breakdown(
            read_model,
            start_date=args.get("start_date"),
            end_date=args.get("end_date"),
            date_field=str(args.get("date_field") or "purchased_at"),
            group_by=str(args.get("group_by") or "store"),
            store_name=args.get("store_name"),
            category=args.get("category"),
            limit=args.get("limit", 100),
        )


class GroceryIntelProductPriceHistoryTool(llm.Tool):
    """Find product price history with scalar parameters."""

    name = "FindProductPriceHistory"
    description = (
        "Find product purchase observations and price history for a product query. "
        "Use this for latest price, price trend, price increases, and store-specific "
        "product history."
    )
    parameters = vol.Schema(
        {
            vol.Required("product_query"): str,
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("start_date"): str,
            vol.Optional("end_date"): str,
            vol.Optional("store_name"): str,
            vol.Optional("limit", default=10): vol.Coerce(int),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Return product price observations."""
        args = tool_input.tool_args
        scope = normalize_scope(args.get("scope"))
        read_model = await _async_read_model(hass, scope)
        return find_product_price_history(
            read_model,
            product_query=str(args.get("product_query") or ""),
            start_date=args.get("start_date"),
            end_date=args.get("end_date"),
            store_name=args.get("store_name"),
            limit=args.get("limit", 10),
        )


class GroceryIntelDataQualityTool(llm.Tool):
    """Inspect common Grocery Intel data quality issues."""

    name = "InspectGroceryDataQuality"
    description = (
        "Inspect common read-model data quality issues such as missing receipt "
        "dates, missing totals, failed extraction, uncategorized receipts, and "
        "low-confidence product matches."
    )
    parameters = vol.Schema(
        {
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("dataset", default="receipts"): vol.In(["receipts", "line_items", "observations"]),
            vol.Optional("issue_type", default="missing_dates"): vol.In(
                ["missing_dates", "missing_totals", "failed_extraction", "uncategorized", "low_confidence"]
            ),
            vol.Optional("limit", default=10): vol.Coerce(int),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Return data quality issues."""
        args = tool_input.tool_args
        scope = normalize_scope(args.get("scope"))
        read_model = await _async_read_model(hass, scope)
        return inspect_grocery_data_quality(
            read_model,
            dataset=str(args.get("dataset") or "receipts"),
            issue_type=str(args.get("issue_type") or "missing_dates"),
            limit=args.get("limit", 10),
        )


class GroceryIntelExportReadModelTool(llm.Tool):
    """Return a capped public read-model export."""

    name = "ExportGroceryIntelDataSnapshot"
    description = (
        "Export a capped read-only Grocery Intel data snapshot. Use this only "
        "when targeted search or analytics calls are not enough and broad local "
        "context is needed. The default analytics scope excludes raw receipt text, "
        "OCR text, local file paths, provider metadata, raw inventory model output, "
        "and fingerprints."
    )
    parameters = vol.Schema(
        {
            vol.Optional("scope", default=ANALYTICS_SCOPE): _scope_schema(),
            vol.Optional("datasets"): vol.Any(list, str),
            vol.Optional("limit_per_dataset", default=DEFAULT_QUERY_LIMIT): vol.Coerce(int),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
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


class GroceryIntelReadModelAPI(llm.API):
    """Grocery Intel read-only public data API for LLM/MCP consumers."""

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """Return the Grocery Intel public read-model API instance."""
        return llm.APIInstance(
            api=self,
            api_prompt=_api_prompt(self.hass, llm_context),
            llm_context=llm_context,
            tools=[
                GroceryIntelDescribeSchemaTool(),
                GroceryIntelSpendSummaryTool(),
                GroceryIntelRecentReceiptsTool(),
                GroceryIntelSpendBreakdownTool(),
                GroceryIntelProductPriceHistoryTool(),
                GroceryIntelDataQualityTool(),
                GroceryIntelQueryTool(),
                GroceryIntelAggregateTool(),
                GroceryIntelExportReadModelTool(),
            ],
        )


@callback
def async_register_llm_api(hass: HomeAssistant) -> Callable[[], None] | None:
    """Register the Grocery Intel LLM API if it is not already registered."""
    if any(api.id == LLM_API_ID for api in llm.async_get_apis(hass)):
        return None
    return llm.async_register_api(
        hass,
        GroceryIntelReadModelAPI(hass=hass, id=LLM_API_ID, name=LLM_API_NAME),
    )


@callback
def async_get_tools(hass: HomeAssistant, llm_context: llm.LLMContext) -> Any:
    """Return Grocery Intel tools for LLM APIs that support contributed tools."""
    try:
        from homeassistant.components import llm as component_llm
    except ImportError:
        return None

    return component_llm.LLMTools(
        tools=[
            GroceryIntelDescribeSchemaTool(),
            GroceryIntelSpendSummaryTool(),
            GroceryIntelRecentReceiptsTool(),
            GroceryIntelSpendBreakdownTool(),
            GroceryIntelProductPriceHistoryTool(),
            GroceryIntelDataQualityTool(),
            GroceryIntelQueryTool(),
            GroceryIntelAggregateTool(),
            GroceryIntelExportReadModelTool(),
        ],
        prompt=_api_prompt(hass, llm_context),
    )
