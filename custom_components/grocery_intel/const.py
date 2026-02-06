"""Constants for Grocery Intel."""

DOMAIN = "grocery_intel"

CONF_CURRENCY_SYMBOL = "currency_symbol"
CONF_OVERPAID_PCT_THRESHOLD = "overpaid_pct_threshold"
CONF_BASELINE_WINDOW_N = "baseline_window_n"
CONF_TOP_INCREASES_RECENT_DAYS = "top_increases_recent_days"
CONF_TOP_INCREASES_PRIOR_DAYS = "top_increases_prior_days"
CONF_BEST_STORE_WINDOW_DAYS = "best_store_window_days"
CONF_RECEIPTS_INBOX_PATH = "receipts_inbox_path"
CONF_RECEIPTS_ARCHIVE_PATH = "receipts_archive_path"
CONF_INBOX_SCAN_INTERVAL_SEC = "inbox_scan_interval_sec"
CONF_ON_SUCCESS = "on_success"
CONF_OCR_ENDPOINT_URL = "ocr_endpoint_url"
CONF_OCR_LANGUAGE = "ocr_language"

DEFAULT_CURRENCY_SYMBOL = "kr"
DEFAULT_OVERPAID_PCT_THRESHOLD = 0.15
DEFAULT_BASELINE_WINDOW_N = 5
DEFAULT_TOP_INCREASES_RECENT_DAYS = 30
DEFAULT_TOP_INCREASES_PRIOR_DAYS = 90
DEFAULT_BEST_STORE_WINDOW_DAYS = 90
DEFAULT_RECEIPTS_INBOX_PATH = "/config/receipts_inbox"
DEFAULT_RECEIPTS_ARCHIVE_PATH = "/config/receipts_archive"
DEFAULT_INBOX_SCAN_INTERVAL_SEC = 300
DEFAULT_ON_SUCCESS = "archive"
DEFAULT_OCR_ENDPOINT_URL = "http://grocery_ocr:8787/v1/ocr"
DEFAULT_OCR_LANGUAGE = "eng"

SERVICE_ADD_RECEIPT = "add_receipt"
SERVICE_UNDO_ACTIVITY = "undo_activity"
SERVICE_REPROCESS_RECEIPTS = "reprocess_receipts"
SERVICE_SCAN_RECEIPTS_INBOX = "scan_receipts_inbox"
SERVICE_RUN_OCR = "run_ocr"

SMALL_OVERPAY_FLOOR = 2.0
