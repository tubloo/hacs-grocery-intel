"""Config flow for Grocery Intel."""
from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries

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
    CONF_INBOX_SCAN_INTERVAL_SEC,
    CONF_ON_SUCCESS,
    CONF_OCR_ENDPOINT_URL,
    CONF_OCR_LANGUAGE,
    DEFAULT_CURRENCY_SYMBOL,
    DEFAULT_OVERPAID_PCT_THRESHOLD,
    DEFAULT_BASELINE_WINDOW_N,
    DEFAULT_TOP_INCREASES_RECENT_DAYS,
    DEFAULT_TOP_INCREASES_PRIOR_DAYS,
    DEFAULT_BEST_STORE_WINDOW_DAYS,
    DEFAULT_RECEIPTS_INBOX_PATH,
    DEFAULT_RECEIPTS_ARCHIVE_PATH,
    DEFAULT_INBOX_SCAN_INTERVAL_SEC,
    DEFAULT_ON_SUCCESS,
    DEFAULT_OCR_ENDPOINT_URL,
    DEFAULT_OCR_LANGUAGE,
)


class GroceryIntelConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Grocery Intel."""

    VERSION = 1

    @staticmethod
    def async_get_options_flow(config_entry):
        return GroceryIntelOptionsFlow(config_entry)

    async def async_step_user(self, user_input=None):
        if user_input is None:
            return self.async_show_form(step_id="user", data_schema=vol.Schema({}))

        return self.async_create_entry(title="Grocery Intel", data={})

    async def async_step_import(self, user_input=None):
        return await self.async_step_user(user_input)


class GroceryIntelOptionsFlow(config_entries.OptionsFlow):
    """Handle options for Grocery Intel."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._entry = config_entry

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_CURRENCY_SYMBOL,
                    default=self._entry.options.get(
                        CONF_CURRENCY_SYMBOL, DEFAULT_CURRENCY_SYMBOL
                    ),
                ): str,
                vol.Optional(
                    CONF_OVERPAID_PCT_THRESHOLD,
                    default=self._entry.options.get(
                        CONF_OVERPAID_PCT_THRESHOLD, DEFAULT_OVERPAID_PCT_THRESHOLD
                    ),
                ): vol.All(float, vol.Range(min=0.01, max=1.0)),
                vol.Optional(
                    CONF_BASELINE_WINDOW_N,
                    default=self._entry.options.get(
                        CONF_BASELINE_WINDOW_N, DEFAULT_BASELINE_WINDOW_N
                    ),
                ): vol.All(int, vol.Range(min=3, max=20)),
                vol.Optional(
                    CONF_TOP_INCREASES_RECENT_DAYS,
                    default=self._entry.options.get(
                        CONF_TOP_INCREASES_RECENT_DAYS,
                        DEFAULT_TOP_INCREASES_RECENT_DAYS,
                    ),
                ): vol.All(int, vol.Range(min=7, max=180)),
                vol.Optional(
                    CONF_TOP_INCREASES_PRIOR_DAYS,
                    default=self._entry.options.get(
                        CONF_TOP_INCREASES_PRIOR_DAYS,
                        DEFAULT_TOP_INCREASES_PRIOR_DAYS,
                    ),
                ): vol.All(int, vol.Range(min=14, max=365)),
                vol.Optional(
                    CONF_BEST_STORE_WINDOW_DAYS,
                    default=self._entry.options.get(
                        CONF_BEST_STORE_WINDOW_DAYS, DEFAULT_BEST_STORE_WINDOW_DAYS
                    ),
                ): vol.All(int, vol.Range(min=30, max=365)),
                vol.Optional(
                    CONF_RECEIPTS_INBOX_PATH,
                    default=self._entry.options.get(
                        CONF_RECEIPTS_INBOX_PATH, DEFAULT_RECEIPTS_INBOX_PATH
                    ),
                ): str,
                vol.Optional(
                    CONF_RECEIPTS_ARCHIVE_PATH,
                    default=self._entry.options.get(
                        CONF_RECEIPTS_ARCHIVE_PATH, DEFAULT_RECEIPTS_ARCHIVE_PATH
                    ),
                ): str,
                vol.Optional(
                    CONF_INBOX_SCAN_INTERVAL_SEC,
                    default=self._entry.options.get(
                        CONF_INBOX_SCAN_INTERVAL_SEC, DEFAULT_INBOX_SCAN_INTERVAL_SEC
                    ),
                ): vol.All(int, vol.Range(min=60, max=3600)),
                vol.Optional(
                    CONF_ON_SUCCESS,
                    default=self._entry.options.get(CONF_ON_SUCCESS, DEFAULT_ON_SUCCESS),
                ): str,
                vol.Optional(
                    CONF_OCR_ENDPOINT_URL,
                    default=self._entry.options.get(
                        CONF_OCR_ENDPOINT_URL, DEFAULT_OCR_ENDPOINT_URL
                    ),
                ): str,
                vol.Optional(
                    CONF_OCR_LANGUAGE,
                    default=self._entry.options.get(CONF_OCR_LANGUAGE, DEFAULT_OCR_LANGUAGE),
                ): str,
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
