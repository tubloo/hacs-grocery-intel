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
    CONF_LLM_PROVIDER,
    CONF_LLM_MODEL,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_EXTRA_INSTRUCTIONS,
    CONF_RECEIPT_CATEGORY_LLM_PROMPT,
    CONF_AZURE_API_VERSION,
    CONF_SHOPPING_AUTO_APPROVE_ENABLED,
    CONF_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
    CONF_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
    CONF_SHOPPING_PAUSE_WHEN_ALL_AWAY,
    CONF_SHOPPING_AUTO_ITEM_MARKER,
    CONF_SHOPPING_AUTO_ITEM_MARKER_POSITION,
    CONF_SHOPPING_TRANSLATE_TO_HA_LANGUAGE,
    CONF_SHOPPING_TRANSLATION_CONFIDENCE_THRESHOLD,
    CONF_INVENTORY_IMAGES_INBOX_PATH,
    CONF_INVENTORY_IMAGES_ARCHIVE_PATH,
    CONF_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS,
    CONF_INVENTORY_IMAGES_SCAN_INTERVAL_SEC,
    CONF_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
    CONF_TELEGRAM_BOT_TOKEN,
    CONF_TELEGRAM_ALLOWED_CHAT_IDS,
    CONF_TELEGRAM_AUTO_DETECT,
    CONF_TELEGRAM_SEND_FEEDBACK,
    CONF_EXPORTS_PATH,
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
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_EXTRA_INSTRUCTIONS,
    DEFAULT_RECEIPT_CATEGORY_LLM_PROMPT,
    DEFAULT_AZURE_API_VERSION,
    DEFAULT_SHOPPING_AUTO_APPROVE_ENABLED,
    DEFAULT_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
    DEFAULT_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
    DEFAULT_SHOPPING_PAUSE_WHEN_ALL_AWAY,
    DEFAULT_SHOPPING_AUTO_ITEM_MARKER,
    DEFAULT_SHOPPING_AUTO_ITEM_MARKER_POSITION,
    DEFAULT_SHOPPING_TRANSLATE_TO_HA_LANGUAGE,
    DEFAULT_SHOPPING_TRANSLATION_CONFIDENCE_THRESHOLD,
    DEFAULT_INVENTORY_IMAGES_INBOX_PATH,
    DEFAULT_INVENTORY_IMAGES_ARCHIVE_PATH,
    DEFAULT_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS,
    DEFAULT_INVENTORY_IMAGES_SCAN_INTERVAL_SEC,
    DEFAULT_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
    DEFAULT_TELEGRAM_BOT_TOKEN,
    DEFAULT_TELEGRAM_ALLOWED_CHAT_IDS,
    DEFAULT_TELEGRAM_AUTO_DETECT,
    DEFAULT_TELEGRAM_SEND_FEEDBACK,
    DEFAULT_EXPORTS_PATH,
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
        self._wizard_stage: int = 0
        self._field_back = "wizard_back"
        self._field_confirm = "wizard_confirm"

    async def async_step_init(self, user_input=None):
        errors: dict[str, str] = {}
        if user_input is not None:
            if self._wizard_stage > 0 and bool(user_input.get(self._field_back, False)):
                self._wizard_stage -= 1
                return self._show_wizard_form(errors)

            self._ingest_stage_input(self._wizard_stage, user_input)
            if self._wizard_stage == 0:
                errors = self._validate_provider_stage()
                if not errors:
                    self._wizard_stage = 1
            elif self._wizard_stage == 1:
                errors = self._validate_connection_stage()
                if not errors:
                    self._wizard_stage = 2
            elif self._wizard_stage == 2:
                self._wizard_stage = 3
            elif self._wizard_stage == 3:
                self._wizard_stage = 4
            elif self._wizard_stage == 4:
                self._wizard_stage = 5
            elif self._wizard_stage == 5:
                self._wizard_stage = 6
            else:
                if not bool(user_input.get(self._field_confirm, False)):
                    errors["base"] = "review_confirm_required"
                    return self._show_wizard_form(errors)
                merged = dict(self._entry.options)
                merged.update(self._user_data)
                merged = self._normalize_provider_options(merged)
                return self.async_create_entry(title="", data=merged)

        return self._show_wizard_form(errors)

    def _show_wizard_form(self, errors: dict[str, str]):
        kwargs = {
            "step_id": "init",
            "data_schema": self._build_schema_for_stage(self._wizard_stage),
            "errors": errors,
        }
        try:
            return self.async_show_form(
                **kwargs,
                description_placeholders=self._stage_placeholders(),
                last_step=(self._wizard_stage == 6),
            )
        except TypeError:
            # Older HA cores may not support newer async_show_form keyword args.
            return self.async_show_form(**kwargs)

    def _opt_default(self, key: str, fallback):
        if key in self._user_data:
            return self._user_data.get(key)
        return self._entry.options.get(key, fallback)

    def _normalize_provider_options(self, options: dict) -> dict:
        """Keep provider-specific auth/endpoint fields explicit and non-stale."""
        out = dict(options)
        provider = str(out.get(CONF_LLM_PROVIDER, "")).strip().lower()

        if provider != "azure":
            out[CONF_AZURE_API_VERSION] = DEFAULT_AZURE_API_VERSION

        if provider == "ollama":
            out[CONF_LLM_API_KEY] = ""

        return out

    def _strip(self, key: str) -> None:
        if key not in self._user_data:
            return
        val = self._user_data.get(key)
        if val is None:
            return
        self._user_data[key] = str(val).strip()

    def _ingest_stage_input(self, stage: int, user_input: dict) -> None:
        filtered = {
            k: v for k, v in user_input.items() if k not in {self._field_back, self._field_confirm}
        }
        self._user_data.update(filtered)
        if stage == 0:
            self._strip(CONF_LLM_PROVIDER)
            return
        if stage == 1:
            for key in (
                CONF_LLM_MODEL,
                CONF_LLM_API_KEY,
                CONF_LLM_BASE_URL,
                CONF_AZURE_API_VERSION,
            ):
                self._strip(key)
            provider = str(self._opt_default(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)).strip().lower()
            if provider == "azure" and not self._user_data.get(CONF_AZURE_API_VERSION):
                self._user_data[CONF_AZURE_API_VERSION] = DEFAULT_AZURE_API_VERSION
            return
        if stage == 2:
            for key in (
                CONF_LLM_EXTRA_INSTRUCTIONS,
                CONF_RECEIPT_CATEGORY_LLM_PROMPT,
            ):
                self._strip(key)
            return
        if stage == 3:
            for key in (
                CONF_RECEIPTS_INBOX_PATH,
                CONF_RECEIPTS_ARCHIVE_PATH,
                CONF_ON_SUCCESS,
            ):
                self._strip(key)
            return
        if stage == 4:
            for key in (
                CONF_INVENTORY_IMAGES_INBOX_PATH,
                CONF_INVENTORY_IMAGES_ARCHIVE_PATH,
            ):
                self._strip(key)
            return
        if stage == 5:
            for key in (
                CONF_TELEGRAM_BOT_TOKEN,
                CONF_TELEGRAM_ALLOWED_CHAT_IDS,
                CONF_SHOPPING_AUTO_ITEM_MARKER,
                CONF_SHOPPING_AUTO_ITEM_MARKER_POSITION,
            ):
                self._strip(key)
            return
        if stage == 6:
            for key in (
                CONF_CURRENCY_SYMBOL,
                CONF_EXPORTS_PATH,
            ):
                self._strip(key)

    def _validate_provider_stage(self) -> dict[str, str]:
        errors: dict[str, str] = {}
        llm_provider = str(self._opt_default(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)).strip().lower()
        if not llm_provider:
            errors["base"] = "llm_provider_required"
        return errors

    def _validate_connection_stage(self) -> dict[str, str]:
        errors: dict[str, str] = {}
        llm_provider = str(self._opt_default(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)).strip().lower()
        llm_model = str(self._opt_default(CONF_LLM_MODEL, DEFAULT_LLM_MODEL)).strip()
        llm_base_url = str(self._opt_default(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL)).strip()

        if not llm_model:
            errors["base"] = "llm_model_required"
        elif llm_provider in {"ollama", "azure"} and not llm_base_url:
            errors["base"] = "llm_base_url_required"
        return errors

    def _build_schema_for_stage(self, stage: int) -> vol.Schema:
        fields: dict = {}

        if stage == 0:
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
            return vol.Schema(fields)

        if stage == 1:
            provider = str(self._opt_default(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)).strip().lower()
            needs_api_key = provider in {"openai", "azure", "google", "anthropic"}
            supports_base_url = provider in {"ollama", "azure", "openai", "google", "anthropic"}

            fields[
                vol.Optional(
                    CONF_LLM_MODEL,
                    default=self._opt_default(CONF_LLM_MODEL, DEFAULT_LLM_MODEL),
                )
            ] = str
            if needs_api_key:
                fields[
                    vol.Optional(
                        CONF_LLM_API_KEY,
                        default=self._opt_default(CONF_LLM_API_KEY, DEFAULT_LLM_API_KEY),
                    )
                ] = selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                )
            if supports_base_url:
                fields[
                    vol.Optional(
                        CONF_LLM_BASE_URL,
                        default=self._opt_default(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL),
                    )
                ] = str
            if provider == "azure":
                fields[
                    vol.Optional(
                        CONF_AZURE_API_VERSION,
                        default=self._opt_default(CONF_AZURE_API_VERSION, DEFAULT_AZURE_API_VERSION),
                    )
                ] = str
            fields[vol.Optional(self._field_back, default=False)] = selector.BooleanSelector()
            return vol.Schema(fields)

        if stage == 2:
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
                    CONF_RECEIPT_CATEGORY_LLM_PROMPT,
                    default=self._opt_default(
                        CONF_RECEIPT_CATEGORY_LLM_PROMPT, DEFAULT_RECEIPT_CATEGORY_LLM_PROMPT
                    ),
                )
            ] = selector.TextSelector(
                selector.TextSelectorConfig(multiline=True)
            )
            fields[vol.Optional(self._field_back, default=False)] = selector.BooleanSelector()
            return vol.Schema(fields)

        if stage == 3:
            fields[
                vol.Optional(
                    CONF_RECEIPTS_INBOX_PATH,
                    default=self._opt_default(CONF_RECEIPTS_INBOX_PATH, DEFAULT_RECEIPTS_INBOX_PATH),
                )
            ] = str
            fields[
                vol.Optional(
                    CONF_RECEIPTS_ARCHIVE_PATH,
                    default=self._opt_default(CONF_RECEIPTS_ARCHIVE_PATH, DEFAULT_RECEIPTS_ARCHIVE_PATH),
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
                    default=self._opt_default(CONF_INBOX_SCAN_INTERVAL_SEC, DEFAULT_INBOX_SCAN_INTERVAL_SEC),
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
            fields[vol.Optional(self._field_back, default=False)] = selector.BooleanSelector()
            return vol.Schema(fields)

        if stage == 4:
            fields[
                vol.Optional(
                    CONF_INVENTORY_IMAGES_INBOX_PATH,
                    default=self._opt_default(
                        CONF_INVENTORY_IMAGES_INBOX_PATH, DEFAULT_INVENTORY_IMAGES_INBOX_PATH
                    ),
                )
            ] = str
            fields[
                vol.Optional(
                    CONF_INVENTORY_IMAGES_ARCHIVE_PATH,
                    default=self._opt_default(
                        CONF_INVENTORY_IMAGES_ARCHIVE_PATH, DEFAULT_INVENTORY_IMAGES_ARCHIVE_PATH
                    ),
                )
            ] = str
            fields[
                vol.Optional(
                    CONF_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS,
                    default=self._opt_default(
                        CONF_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS, DEFAULT_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS
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
                        CONF_INVENTORY_IMAGES_SCAN_INTERVAL_SEC, DEFAULT_INVENTORY_IMAGES_SCAN_INTERVAL_SEC
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
                        CONF_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS, DEFAULT_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS
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
            fields[vol.Optional(self._field_back, default=False)] = selector.BooleanSelector()
            return vol.Schema(fields)

        if stage == 5:
            fields[
                vol.Optional(
                    CONF_SHOPPING_AUTO_APPROVE_ENABLED,
                    default=self._opt_default(
                        CONF_SHOPPING_AUTO_APPROVE_ENABLED, DEFAULT_SHOPPING_AUTO_APPROVE_ENABLED
                    ),
                )
            ] = selector.BooleanSelector()
            fields[
                vol.Optional(
                    CONF_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
                    default=self._opt_default(
                        CONF_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS, DEFAULT_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS
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
                    default=self._opt_default(CONF_SHOPPING_PAUSE_WHEN_ALL_AWAY, DEFAULT_SHOPPING_PAUSE_WHEN_ALL_AWAY),
                )
            ] = selector.BooleanSelector()
            fields[
                vol.Optional(
                    CONF_SHOPPING_TRANSLATE_TO_HA_LANGUAGE,
                    default=self._opt_default(
                        CONF_SHOPPING_TRANSLATE_TO_HA_LANGUAGE,
                        DEFAULT_SHOPPING_TRANSLATE_TO_HA_LANGUAGE,
                    ),
                )
            ] = selector.BooleanSelector()
            fields[
                vol.Optional(
                    CONF_SHOPPING_TRANSLATION_CONFIDENCE_THRESHOLD,
                    default=self._opt_default(
                        CONF_SHOPPING_TRANSLATION_CONFIDENCE_THRESHOLD,
                        DEFAULT_SHOPPING_TRANSLATION_CONFIDENCE_THRESHOLD,
                    ),
                )
            ] = vol.All(float, vol.Range(min=0.5, max=0.99))
            fields[
                vol.Optional(
                    CONF_SHOPPING_AUTO_ITEM_MARKER,
                    default=self._opt_default(CONF_SHOPPING_AUTO_ITEM_MARKER, DEFAULT_SHOPPING_AUTO_ITEM_MARKER),
                )
            ] = str
            fields[
                vol.Optional(
                    CONF_SHOPPING_AUTO_ITEM_MARKER_POSITION,
                    default=self._opt_default(
                        CONF_SHOPPING_AUTO_ITEM_MARKER_POSITION,
                        DEFAULT_SHOPPING_AUTO_ITEM_MARKER_POSITION,
                    ),
                )
            ] = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        {"value": "prefix", "label": "Prefix"},
                        {"value": "suffix", "label": "Suffix"},
                    ],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
            fields[
                vol.Optional(
                    CONF_TELEGRAM_BOT_TOKEN,
                    default=self._opt_default(CONF_TELEGRAM_BOT_TOKEN, DEFAULT_TELEGRAM_BOT_TOKEN),
                )
            ] = selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD))
            fields[
                vol.Optional(
                    CONF_TELEGRAM_ALLOWED_CHAT_IDS,
                    default=self._opt_default(CONF_TELEGRAM_ALLOWED_CHAT_IDS, DEFAULT_TELEGRAM_ALLOWED_CHAT_IDS),
                )
            ] = str
            fields[
                vol.Optional(
                    CONF_TELEGRAM_AUTO_DETECT,
                    default=self._opt_default(CONF_TELEGRAM_AUTO_DETECT, DEFAULT_TELEGRAM_AUTO_DETECT),
                )
            ] = selector.BooleanSelector()
            fields[
                vol.Optional(
                    CONF_TELEGRAM_SEND_FEEDBACK,
                    default=self._opt_default(CONF_TELEGRAM_SEND_FEEDBACK, DEFAULT_TELEGRAM_SEND_FEEDBACK),
                )
            ] = selector.BooleanSelector()
            fields[vol.Optional(self._field_back, default=False)] = selector.BooleanSelector()
            return vol.Schema(fields)

        fields[
            vol.Optional(
                CONF_CURRENCY_SYMBOL,
                default=self._opt_default(CONF_CURRENCY_SYMBOL, DEFAULT_CURRENCY_SYMBOL),
            )
        ] = str
        fields[
            vol.Optional(
                CONF_OVERPAID_PCT_THRESHOLD,
                default=self._opt_default(CONF_OVERPAID_PCT_THRESHOLD, DEFAULT_OVERPAID_PCT_THRESHOLD),
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
                default=self._opt_default(CONF_TOP_INCREASES_RECENT_DAYS, DEFAULT_TOP_INCREASES_RECENT_DAYS),
            )
        ] = vol.All(int, vol.Range(min=7, max=180))
        fields[
            vol.Optional(
                CONF_TOP_INCREASES_PRIOR_DAYS,
                default=self._opt_default(CONF_TOP_INCREASES_PRIOR_DAYS, DEFAULT_TOP_INCREASES_PRIOR_DAYS),
            )
        ] = vol.All(int, vol.Range(min=14, max=365))
        fields[
            vol.Optional(
                CONF_BEST_STORE_WINDOW_DAYS,
                default=self._opt_default(CONF_BEST_STORE_WINDOW_DAYS, DEFAULT_BEST_STORE_WINDOW_DAYS),
            )
        ] = vol.All(int, vol.Range(min=30, max=365))
        fields[
            vol.Optional(
                CONF_EXPORTS_PATH,
                default=self._opt_default(CONF_EXPORTS_PATH, DEFAULT_EXPORTS_PATH),
            )
        ] = str
        fields[vol.Optional(self._field_back, default=False)] = selector.BooleanSelector()
        fields[vol.Required(self._field_confirm, default=False)] = selector.BooleanSelector()
        return vol.Schema(fields)

    def _stage_placeholders(self) -> dict[str, str]:
        labels = {
            0: "LLM Provider",
            1: "LLM Connection",
            2: "LLM Prompting",
            3: "Receipts",
            4: "Inventory",
            5: "Automation",
            6: "Review",
        }
        current = int(self._wizard_stage) + 1
        total = 7
        return {
            "wizard_step_current": str(current),
            "wizard_step_total": str(total),
            "wizard_stage_label": labels.get(self._wizard_stage, "Options"),
        }
