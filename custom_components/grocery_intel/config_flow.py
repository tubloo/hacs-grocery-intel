"""Config flow for Grocery Intel."""
from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_CURRENCY_SYMBOL,
    CONF_OVERPAID_PCT_THRESHOLD,
    CONF_BASELINE_WINDOW_N,
    CONF_TOP_INCREASES_RECENT_DAYS,
    CONF_TOP_INCREASES_PRIOR_DAYS,
    CONF_BEST_STORE_WINDOW_DAYS,
    CONF_RECEIPTS_INBOX_PATH,
    CONF_RECEIPTS_ARCHIVE_PATH,
    CONF_RECEIPTS_ARCHIVE_TTL_DAYS,
    CONF_INBOX_SCAN_INTERVAL_SEC,
    CONF_ON_SUCCESS,
    CONF_OCR_ENDPOINT_URL,
    CONF_OCR_LANGUAGE,
    CONF_OCR_API_TOKEN,
    CONF_OCR_API_TOKEN_HEADER,
    CONF_EXTRACTOR_MODE,
    CONF_LLM_PROVIDER,
    CONF_LLM_MODEL,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_EXTRA_INSTRUCTIONS,
    CONF_AZURE_API_VERSION,
    CONF_SHOPPING_AUTO_APPROVE_ENABLED,
    CONF_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
    CONF_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
    CONF_SHOPPING_PAUSE_WHEN_ALL_AWAY,
    CONF_INVENTORY_IMAGES_INBOX_PATH,
    CONF_INVENTORY_IMAGES_ARCHIVE_PATH,
    CONF_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS,
    CONF_INVENTORY_IMAGES_SCAN_INTERVAL_SEC,
    CONF_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
    DEFAULT_CURRENCY_SYMBOL,
    DEFAULT_OVERPAID_PCT_THRESHOLD,
    DEFAULT_BASELINE_WINDOW_N,
    DEFAULT_TOP_INCREASES_RECENT_DAYS,
    DEFAULT_TOP_INCREASES_PRIOR_DAYS,
    DEFAULT_BEST_STORE_WINDOW_DAYS,
    DEFAULT_RECEIPTS_INBOX_PATH,
    DEFAULT_RECEIPTS_ARCHIVE_PATH,
    DEFAULT_RECEIPTS_ARCHIVE_TTL_DAYS,
    DEFAULT_INBOX_SCAN_INTERVAL_SEC,
    DEFAULT_ON_SUCCESS,
    DEFAULT_OCR_ENDPOINT_URL,
    DEFAULT_OCR_LANGUAGE,
    DEFAULT_OCR_API_TOKEN,
    DEFAULT_OCR_API_TOKEN_HEADER,
    DEFAULT_EXTRACTOR_MODE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_EXTRA_INSTRUCTIONS,
    DEFAULT_AZURE_API_VERSION,
    DEFAULT_SHOPPING_AUTO_APPROVE_ENABLED,
    DEFAULT_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
    DEFAULT_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
    DEFAULT_SHOPPING_PAUSE_WHEN_ALL_AWAY,
    DEFAULT_INVENTORY_IMAGES_INBOX_PATH,
    DEFAULT_INVENTORY_IMAGES_ARCHIVE_PATH,
    DEFAULT_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS,
    DEFAULT_INVENTORY_IMAGES_SCAN_INTERVAL_SEC,
    DEFAULT_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
)


class GroceryIntelConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Grocery Intel."""

    VERSION = 1
    _UNIQUE_ID = "grocery_intel"

    @staticmethod
    def async_get_options_flow(config_entry):
        return GroceryIntelOptionsFlow(config_entry)

    async def async_step_user(self, user_input=None):
        await self.async_set_unique_id(self._UNIQUE_ID)
        self._abort_if_unique_id_configured()
        if user_input is None:
            return self.async_show_form(step_id="user", data_schema=vol.Schema({}))

        return self.async_create_entry(title="Grocery Intel", data={})

    async def async_step_import(self, user_input=None):
        return await self.async_step_user(user_input)


class GroceryIntelOptionsFlow(config_entries.OptionsFlow):
    """Handle options for Grocery Intel."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._entry = config_entry
        self._user_data: dict = {}
        self._form_mode: str = self._entry.options.get(
            CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE
        )

    async def async_step_init(self, user_input=None):
        errors: dict[str, str] = {}
        if user_input is not None:
            self._user_data.update(user_input)

            mode = (self._user_data.get(CONF_EXTRACTOR_MODE) or DEFAULT_EXTRACTOR_MODE).strip()
            self._user_data[CONF_EXTRACTOR_MODE] = mode

            def _strip(key: str) -> None:
                if key not in self._user_data:
                    return
                val = self._user_data.get(key)
                if val is None:
                    return
                self._user_data[key] = str(val).strip()

            _strip(CONF_OCR_ENDPOINT_URL)
            _strip(CONF_OCR_API_TOKEN)
            _strip(CONF_LLM_PROVIDER)
            _strip(CONF_LLM_MODEL)
            _strip(CONF_LLM_API_KEY)
            _strip(CONF_LLM_BASE_URL)
            _strip(CONF_LLM_EXTRA_INSTRUCTIONS)
            _strip(CONF_AZURE_API_VERSION)

            if not self._user_data.get(CONF_AZURE_API_VERSION):
                self._user_data[CONF_AZURE_API_VERSION] = DEFAULT_AZURE_API_VERSION

            ocr_url = (self._user_data.get(CONF_OCR_ENDPOINT_URL) or "").strip()
            llm_provider = (self._user_data.get(CONF_LLM_PROVIDER) or "").strip()
            llm_model = (self._user_data.get(CONF_LLM_MODEL) or "").strip()
            llm_base_url = (self._user_data.get(CONF_LLM_BASE_URL) or "").strip()

            if mode in {"heuristic", "hybrid"} and not ocr_url:
                errors["base"] = "ocr_endpoint_required"
            elif mode in {"llm", "hybrid"}:
                if not llm_provider:
                    errors["base"] = "llm_provider_required"
                elif not llm_model:
                    errors["base"] = "llm_model_required"
                elif llm_provider in {"ollama", "azure"} and not llm_base_url:
                    errors["base"] = "llm_base_url_required"

            if not errors:
                merged = dict(self._entry.options)
                merged.update(self._user_data)
                return self.async_create_entry(title="", data=merged)

        # Show the form based on the last selected/saved mode.
        mode = (
            (self._user_data.get(CONF_EXTRACTOR_MODE) or self._form_mode or DEFAULT_EXTRACTOR_MODE)
            .strip()
        )
        self._form_mode = mode
        return self.async_show_form(
            step_id="init",
            data_schema=self._build_schema(mode),
            errors=errors,
        )

    def _opt_default(self, key: str, fallback):
        if key in self._user_data:
            return self._user_data.get(key)
        return self._entry.options.get(key, fallback)

    def _build_schema(self, mode: str) -> vol.Schema:
        fields: dict = {}

        # Always show OCR/LLM settings so users can pre-configure them.
        # Runtime behavior is still controlled by extractor_mode.
        show_ocr = True
        show_llm = True

        # Extractor mode first (drives which fields are shown)
        fields[
            vol.Optional(
                CONF_EXTRACTOR_MODE,
                default=self._opt_default(CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE),
            )
        ] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=["heuristic", "llm", "hybrid"],
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        )

        # OCR fields
        if show_ocr:
            fields[
                vol.Optional(
                    CONF_OCR_ENDPOINT_URL,
                    default=self._opt_default(CONF_OCR_ENDPOINT_URL, DEFAULT_OCR_ENDPOINT_URL),
                )
            ] = str
            fields[
                vol.Optional(
                    CONF_OCR_LANGUAGE,
                    default=self._opt_default(CONF_OCR_LANGUAGE, DEFAULT_OCR_LANGUAGE),
                )
            ] = str
            fields[
                vol.Optional(
                    CONF_OCR_API_TOKEN,
                    default=self._opt_default(CONF_OCR_API_TOKEN, DEFAULT_OCR_API_TOKEN),
                )
            ] = str
            fields[
                vol.Optional(
                    CONF_OCR_API_TOKEN_HEADER,
                    default=self._opt_default(
                        CONF_OCR_API_TOKEN_HEADER, DEFAULT_OCR_API_TOKEN_HEADER
                    ),
                )
            ] = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=["Authorization", "X-API-Key", "api-key"],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )

        # LLM fields
        if show_llm:
            fields[
                vol.Optional(
                    CONF_LLM_PROVIDER,
                    default=self._opt_default(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER),
                )
            ] = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        {"value": "openai", "label": "OpenAI"},
                        {"value": "azure", "label": "Azure OpenAI"},
                        {"value": "google", "label": "Google"},
                        {"value": "anthropic", "label": "Anthropic"},
                        {"value": "ollama", "label": "Ollama"},
                    ],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
            fields[
                vol.Optional(
                    CONF_LLM_MODEL,
                    default=self._opt_default(CONF_LLM_MODEL, DEFAULT_LLM_MODEL),
                )
            ] = str
            fields[
                vol.Optional(
                    CONF_LLM_API_KEY,
                    default=self._opt_default(CONF_LLM_API_KEY, DEFAULT_LLM_API_KEY),
                )
            ] = selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
            )
            fields[
                vol.Optional(
                    CONF_LLM_BASE_URL,
                    default=self._opt_default(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL),
                )
            ] = str
            fields[
                vol.Optional(
                    CONF_LLM_EXTRA_INSTRUCTIONS,
                    default=self._opt_default(
                        CONF_LLM_EXTRA_INSTRUCTIONS, DEFAULT_LLM_EXTRA_INSTRUCTIONS
                    ),
                )
            ] = selector.TextSelector(
                selector.TextSelectorConfig(multiline=True)
            )
            fields[
                vol.Optional(
                    CONF_AZURE_API_VERSION,
                    default=self._opt_default(CONF_AZURE_API_VERSION, DEFAULT_AZURE_API_VERSION),
                )
            ] = str

        # Remaining options (always shown)
        fields[
            vol.Optional(
                CONF_CURRENCY_SYMBOL,
                default=self._opt_default(CONF_CURRENCY_SYMBOL, DEFAULT_CURRENCY_SYMBOL),
            )
        ] = str
        fields[
            vol.Optional(
                CONF_OVERPAID_PCT_THRESHOLD,
                default=self._opt_default(
                    CONF_OVERPAID_PCT_THRESHOLD, DEFAULT_OVERPAID_PCT_THRESHOLD
                ),
            )
        ] = vol.All(float, vol.Range(min=0.01, max=1.0))
        fields[
            vol.Optional(
                CONF_BASELINE_WINDOW_N,
                default=self._opt_default(CONF_BASELINE_WINDOW_N, DEFAULT_BASELINE_WINDOW_N),
            )
        ] = vol.All(int, vol.Range(min=3, max=20))
        fields[
            vol.Optional(
                CONF_TOP_INCREASES_RECENT_DAYS,
                default=self._opt_default(
                    CONF_TOP_INCREASES_RECENT_DAYS, DEFAULT_TOP_INCREASES_RECENT_DAYS
                ),
            )
        ] = vol.All(int, vol.Range(min=7, max=180))
        fields[
            vol.Optional(
                CONF_TOP_INCREASES_PRIOR_DAYS,
                default=self._opt_default(
                    CONF_TOP_INCREASES_PRIOR_DAYS, DEFAULT_TOP_INCREASES_PRIOR_DAYS
                ),
            )
        ] = vol.All(int, vol.Range(min=14, max=365))
        fields[
            vol.Optional(
                CONF_BEST_STORE_WINDOW_DAYS,
                default=self._opt_default(
                    CONF_BEST_STORE_WINDOW_DAYS, DEFAULT_BEST_STORE_WINDOW_DAYS
                ),
            )
        ] = vol.All(int, vol.Range(min=30, max=365))
        fields[
            vol.Optional(
                CONF_RECEIPTS_INBOX_PATH,
                default=self._opt_default(
                    CONF_RECEIPTS_INBOX_PATH, DEFAULT_RECEIPTS_INBOX_PATH
                ),
            )
        ] = str
        fields[
            vol.Optional(
                CONF_RECEIPTS_ARCHIVE_PATH,
                default=self._opt_default(
                    CONF_RECEIPTS_ARCHIVE_PATH, DEFAULT_RECEIPTS_ARCHIVE_PATH
                ),
            )
        ] = str
        fields[
            vol.Optional(
                CONF_RECEIPTS_ARCHIVE_TTL_DAYS,
                default=self._opt_default(
                    CONF_RECEIPTS_ARCHIVE_TTL_DAYS, DEFAULT_RECEIPTS_ARCHIVE_TTL_DAYS
                ),
            )
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1,
                max=90,
                step=1,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="d",
            )
        )
        fields[
            vol.Optional(
                CONF_INBOX_SCAN_INTERVAL_SEC,
                default=self._opt_default(
                    CONF_INBOX_SCAN_INTERVAL_SEC, DEFAULT_INBOX_SCAN_INTERVAL_SEC
                ),
            )
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=30,
                max=600,
                step=10,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="s",
            )
        )
        fields[
            vol.Optional(
                CONF_ON_SUCCESS,
                default=self._opt_default(CONF_ON_SUCCESS, DEFAULT_ON_SUCCESS),
            )
        ] = str

        fields[
            vol.Optional(
                CONF_SHOPPING_AUTO_APPROVE_ENABLED,
                default=self._opt_default(
                    CONF_SHOPPING_AUTO_APPROVE_ENABLED,
                    DEFAULT_SHOPPING_AUTO_APPROVE_ENABLED,
                ),
            )
        ] = selector.BooleanSelector()

        fields[
            vol.Optional(
                CONF_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
                default=self._opt_default(
                    CONF_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
                    DEFAULT_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
                ),
            )
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1,
                max=30,
                step=1,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="d",
            )
        )

        fields[
            vol.Optional(
                CONF_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
                default=self._opt_default(
                    CONF_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
                    DEFAULT_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
                ),
            )
        ] = vol.All(float, vol.Range(min=0.5, max=0.99))

        fields[
            vol.Optional(
                CONF_SHOPPING_PAUSE_WHEN_ALL_AWAY,
                default=self._opt_default(
                    CONF_SHOPPING_PAUSE_WHEN_ALL_AWAY,
                    DEFAULT_SHOPPING_PAUSE_WHEN_ALL_AWAY,
                ),
            )
        ] = selector.BooleanSelector()

        fields[
            vol.Optional(
                CONF_INVENTORY_IMAGES_INBOX_PATH,
                default=self._opt_default(
                    CONF_INVENTORY_IMAGES_INBOX_PATH,
                    DEFAULT_INVENTORY_IMAGES_INBOX_PATH,
                ),
            )
        ] = str

        fields[
            vol.Optional(
                CONF_INVENTORY_IMAGES_ARCHIVE_PATH,
                default=self._opt_default(
                    CONF_INVENTORY_IMAGES_ARCHIVE_PATH,
                    DEFAULT_INVENTORY_IMAGES_ARCHIVE_PATH,
                ),
            )
        ] = str

        fields[
            vol.Optional(
                CONF_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS,
                default=self._opt_default(
                    CONF_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS,
                    DEFAULT_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS,
                ),
            )
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1,
                max=90,
                step=1,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="d",
            )
        )

        fields[
            vol.Optional(
                CONF_INVENTORY_IMAGES_SCAN_INTERVAL_SEC,
                default=self._opt_default(
                    CONF_INVENTORY_IMAGES_SCAN_INTERVAL_SEC,
                    DEFAULT_INVENTORY_IMAGES_SCAN_INTERVAL_SEC,
                ),
            )
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=30,
                max=3600,
                step=30,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="s",
            )
        )

        fields[
            vol.Optional(
                CONF_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
                default=self._opt_default(
                    CONF_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
                    DEFAULT_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
                ),
            )
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1,
                max=30,
                step=1,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="d",
            )
        )

        return vol.Schema(fields)
