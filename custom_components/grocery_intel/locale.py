"""Locale-aware parsing profiles.

This integration is intended to work globally without shipping a country DB.
Instead, we keep small, composable locale profiles (language + country) that
provide keywords/hints for parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from homeassistant.core import HomeAssistant


@dataclass(frozen=True, slots=True)
class LocaleProfile:
    language: str  # e.g. "en", "sv", "es"
    country: str  # e.g. "US", "SE", "ES", "CO"

    # Total keywords
    total_strong: tuple[str, ...]
    total_weak: tuple[str, ...]
    total_negative: tuple[str, ...]

    # Date keywords (helps weigh candidates)
    date_positive: tuple[str, ...]
    date_negative: tuple[str, ...]

    # Store parsing hints
    store_stop_exact: frozenset[str]
    store_stop_contains: tuple[str, ...]
    store_address_hints: tuple[str, ...]
    store_brand_hints: tuple[str, ...]

    # Month name map additions (lowercased)
    month_map: dict[str, int]


def _base_month_map() -> dict[str, int]:
    # Common English (baseline). Locale packs extend this.
    return {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }


DEFAULT_MONTH_MAP: Final[dict[str, int]] = _base_month_map()


def _base_profile(*, language: str, country: str) -> LocaleProfile:
    stop_exact = frozenset(
        {
            "receipt",
            "store",
            "total",
            "subtotal",
            "date",
            "time",
            "thanks",
            "welcome",
            "cashier",
        }
    )

    return LocaleProfile(
        language=language,
        country=country,
        total_strong=("grand total", "total to pay", "amount due"),
        total_weak=("total", "amount", "balance", "grand"),
        total_negative=("subtotal",),
        date_positive=("date", "time", "purchase", "cashier"),
        date_negative=("expiry", "best before", "due"),
        store_stop_exact=stop_exact,
        store_stop_contains=(
            "vat",
            "tax",
            "phone",
            "tel",
            "www.",
            "http",
            "address",
            "terminal",
        ),
        store_address_hints=(
            "street",
            "st ",
            "road",
            "rd ",
            "ave",
            "avenue",
            "postcode",
            "zip",
        ),
        store_brand_hints=(),
        month_map=_base_month_map(),
    )


def _sv_profile() -> LocaleProfile:
    base = _base_profile(language="sv", country="SE")
    month_map = dict(base.month_map)
    month_map.update(
        {
            "januari": 1,
            "februari": 2,
            "mars": 3,
            "maj": 5,
            "juni": 6,
            "juli": 7,
            "augusti": 8,
            "september": 9,
            "okt": 10,
            "oktober": 10,
            "november": 11,
            "dek": 12,
            "december": 12,
        }
    )
    return LocaleProfile(
        language="sv",
        country="SE",
        total_strong=("att betala", "attbetala", "summa att betala", "totalt att betala"),
        total_weak=("summa", "total", "totalt", "belopp"),
        total_negative=("delsumma", "subtotal"),
        date_positive=("datum", "tid", "köp", "kop", "kassa"),
        date_negative=("bäst före", "bast fore", "förfall", "forfall"),
        store_stop_exact=frozenset(
            set(base.store_stop_exact)
            | {"kvitto", "butik", "summa", "datum", "tid", "tack", "välkommen"}
        ),
        store_stop_contains=base.store_stop_contains
        + ("org.nr", "org nr", "moms", "telefon", "kvitt", "kassa"),
        store_address_hints=base.store_address_hints
        + ("gata", "gatan", "väg", "vagen", "vägen", "postnr", "postnummer"),
        # Keep Sweden-specific chain hints here (scalable by adding more locale packs).
        store_brand_hints=(
            "ica",
            "coop",
            "willys",
            "hemköp",
            "city gross",
            "lidl",
            "maxi",
            "7-eleven",
            "seven eleven",
            "systembolaget",
        ),
        month_map=month_map,
    )


def _es_es_profile() -> LocaleProfile:
    base = _base_profile(language="es", country="ES")
    month_map = dict(base.month_map)
    month_map.update(
        {
            "ene": 1,
            "enero": 1,
            "feb": 2,
            "febrero": 2,
            "mar": 3,
            "marzo": 3,
            "abr": 4,
            "abril": 4,
            "may": 5,
            "mayo": 5,
            "jun": 6,
            "junio": 6,
            "jul": 7,
            "julio": 7,
            "ago": 8,
            "agosto": 8,
            "sep": 9,
            "sept": 9,
            "septiembre": 9,
            "oct": 10,
            "octubre": 10,
            "nov": 11,
            "noviembre": 11,
            "dic": 12,
            "diciembre": 12,
        }
    )
    return LocaleProfile(
        language="es",
        country="ES",
        total_strong=("total a pagar", "importe total", "a pagar"),
        total_weak=("total", "importe", "saldo"),
        total_negative=("subtotal",),
        date_positive=("fecha", "hora", "compra"),
        date_negative=("caducidad", "venc", "consumir preferentemente"),
        store_stop_exact=frozenset(set(base.store_stop_exact) | {"ticket", "gracias", "bienvenido"}),
        store_stop_contains=base.store_stop_contains + ("cif", "nif", "iva", "teléfono", "telefono"),
        store_address_hints=base.store_address_hints
        + ("calle", "c/", "av", "avenida", "plaza", "paseo", "cp"),
        store_brand_hints=(),
        month_map=month_map,
    )


def _es_co_profile() -> LocaleProfile:
    # Colombian Spanish differences: common tax id label "NIT" and address formats.
    base = _es_es_profile()
    return LocaleProfile(
        language="es",
        country="CO",
        total_strong=base.total_strong + ("total pagar",),
        total_weak=base.total_weak,
        total_negative=base.total_negative,
        date_positive=base.date_positive,
        date_negative=base.date_negative,
        store_stop_exact=base.store_stop_exact,
        store_stop_contains=base.store_stop_contains + ("nit", "régimen", "regimen"),
        store_address_hints=base.store_address_hints + ("carrera", "cra", "cll", "km", "bogotá", "medellín"),
        store_brand_hints=base.store_brand_hints,
        month_map=dict(base.month_map),
    )


PROFILES: Final[dict[tuple[str, str], LocaleProfile]] = {
    ("sv", "SE"): _sv_profile(),
    ("es", "ES"): _es_es_profile(),
    ("es", "CO"): _es_co_profile(),
}


def get_locale_profile(hass: HomeAssistant) -> LocaleProfile:
    """Pick the best locale profile for this HA instance.

    - Prefer full (language,country) match.
    - Fallback to language-only (any country).
    - Fallback to base (en,US) profile.
    """
    lang = (getattr(hass.config, "language", "") or "").split("-")[0].lower() or "en"
    country = (getattr(hass.config, "country", "") or "").upper() or "US"

    # 1) Exact language-country match
    prof = PROFILES.get((lang, country))
    if prof:
        return prof

    # 2) Language-only match (any country). Useful when receipts follow UI language.
    for (plang, _pcountry), p in PROFILES.items():
        if plang == lang:
            return p

    # 3) Country-only match (any language). Useful when HA UI is English but home country is not,
    # and receipts are typically in the local language.
    for (_plang, pcountry), p in PROFILES.items():
        if pcountry == country:
            return p

    # 4) Global baseline
    return _base_profile(language="en", country=country)
