import os
import json
import logging
import warnings
from base64 import b64encode
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode
from typing import Any, Dict, Iterable, Mapping, Optional
from html import escape
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    insert,
    select,
    update,
    func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, NoSuchTableError
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode, DataReturnMode

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore


logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv(override=False)

REMINDER_LABEL_SELECTORS = {
    "Reminder Recipient": [".st-key-reminder_recipient_c label"],
    "Reminder Date": [".st-key-reminder_date_c label"],
    "Reminder Time": [".st-key-reminder_time_c label"],
    "Reminder Note": [".st-key-reminder_note_c label"],
}

REMINDER_ALL_VALUE = "All"

st.set_page_config(page_title="CALL_TICKET_UNIFY", layout="wide")

APP_TIMEZONE = ZoneInfo("Asia/Baghdad")
UTC_TIMEZONE = ZoneInfo("UTC")


def _now_local() -> datetime:
    """Return the current local time in Baghdad as a naive datetime."""
    return datetime.now(APP_TIMEZONE).replace(tzinfo=None)


def _to_local_naive(dt: datetime, *, assume_utc_if_naive: bool = False) -> datetime:
    """Normalize a datetime to the Baghdad timezone and strip tzinfo.

    Parameters
    ----------
    dt:
        Input datetime to convert.
    assume_utc_if_naive:
        When True, naive datetimes are interpreted as UTC before conversion.
    """

    if dt.tzinfo is None:
        if assume_utc_if_naive:
            dt = dt.replace(tzinfo=UTC_TIMEZONE)
        else:
            return dt.replace(tzinfo=None)

    localized = dt.astimezone(APP_TIMEZONE)
    return localized.replace(tzinfo=None)

# Silence third-party FutureWarning spam from st_aggrid until the library upgrades its pandas usage.
warnings.filterwarnings(
    "ignore",
    message=r"DataFrame\.applymap has been deprecated",
    category=FutureWarning,
    module="st_aggrid",
)


def _configure_logging() -> None:
    level_name = os.getenv("CALL_TICKET_LOG_LEVEL", "INFO").upper()
    if level_name not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        level_name = "INFO"

    level_value = getattr(logging, level_name, logging.INFO)

    try:
        st.set_option("logger.level", level_name.lower())
    except Exception:
        pass

    try:
        session_state = st.session_state
    except Exception:
        session_state = None

    already_configured = False
    if session_state is not None:
        already_configured = bool(session_state.get("_tickets_logger_configured"))

    root_logger = logging.getLogger()
    if not already_configured and not root_logger.handlers:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            level=level_value,
        )
    root_logger.setLevel(level_value)
    logger.setLevel(level_value)

    if session_state is not None and not already_configured:
        session_state["_tickets_logger_configured"] = True


_configure_logging()

# --- Database configuration ---
DATABASE_URL_ENV_KEYS = ("NEON_DATABASE_URL", "DATABASE_URL", "POSTGRES_URL")
LOCAL_SECRETS_FILENAME = "local_secrets.toml"
LOCAL_SECRETS_PATH = Path(__file__).with_name(LOCAL_SECRETS_FILENAME)

TICKET_COLUMN_ORDER = [
    "id",
    "created_at",
    "ticket_group",
    "ont_id",
    "call_type",
    "description",
    "activity_inquiry_type",
    "digicare_issue_type",
    "complaint_type",
    "refund_type",
    "employee_suggestion",
    "device_location",
    "root_cause",
    "ont_model",
    "complaint_status",
    "kurdtel_service_status",
    "osp_type",
    "city",
    "issue_type",
    "fttg",
    "olt",
    "second_number",
    "created_by",
    "address",
    "outage_start_date",
    "outage_end_date",
    "fttx_job_status",
    "fttx_job_remarks",
    "fttx_cancel_reason",
    "callback_status",
    "callback_reason",
    "followup_status",
    "online_game",
    "ip",
    "vlan",
    "packet_loss",
    "high_ping",
    "visit_required",
    "reminder_enabled",
    "reminder_recipient",
    "reminder_note",
    "reminder_at",
    "channel",
]

BOOLEAN_COLUMNS = {"reminder_enabled", "visit_required"}
DATE_COLUMNS = {"outage_start_date", "outage_end_date"}
READ_ONLY_DETAIL_FIELDS = {"id", "created_at"}
DETAIL_MULTILINE_FIELDS = {
    "description",
    "address",
    "reminder_note",
    "fttx_job_remarks",
    "fttx_job_status",
    "followup_status",
    "callback_reason",
    "callback_status",
}

GRID_DEFAULT_DISPLAY_COLUMNS = [
    "id", "created_at", "ticket_group", "ticket_type", "ont_id", "call_type",
    "created_by", "complaint_status", "employee_suggestion", "device_location", "root_cause",
    "ont_model", "olt", "callback_status", "callback_reason", "followup_status",
    "issue_type", "fttg", "second_number", "fttx_job_status", "fttx_cancel_reason", "fttx_job_remarks",
    "refund_type", "online_game", "kurdtel_service_status", "city", "address", "description",
    "visit_required", "reminder_enabled", "reminder_recipient", "reminder_note", "reminder_at", "channel",
]

GRID_REQUIRED_COLUMNS = ["id"]

TICKET_GROUP_BADGE_COLORS = {
    "activities & inquiries": "#2563eb",
    "complaints": "#dc2626",
    "osp appointments": "#7c3aed",
}

COMPLAINT_STATUS_BADGE_COLORS = {
    "solved": "#16a34a",
    "not solved": "#dc2626",
    "pending": "#f59e0b",
}


def _resolve_badge_color(mapping: dict[str, str], value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return mapping.get(text)


def _make_badge_renderer_js(class_name: str, color_map: dict[str, str], *, default_color: str = "#4b5563") -> JsCode:
    normalized = {str(key).lower(): value for key, value in color_map.items() if value}
    mapping_json = json.dumps(normalized)
    return JsCode(
        f"""
        class {class_name} {{
            init(params){{
                const raw = (params && params.value) != null ? params.value : '';
                const text = raw === null || raw === undefined ? '' : String(raw);
                if (!text) {{
                    this.eGui = document.createTextNode('');
                    return;
                }}
                const key = text.toLowerCase();
                const colorMap = {mapping_json};
                const color = colorMap[key] || '{default_color}';
                const el = document.createElement('span');
                el.textContent = text;
                el.className = 'grid-badge';
                el.style.display = 'inline-flex';
                el.style.alignItems = 'center';
                el.style.justifyContent = 'center';
                el.style.padding = '2px 8px';
                el.style.borderRadius = '999px';
                el.style.fontSize = '12px';
                el.style.fontWeight = '600';
                el.style.lineHeight = '1.2';
                el.style.whiteSpace = 'nowrap';
                el.style.backgroundColor = color;
                el.style.color = '#ffffff';
                this.eGui = el;
            }}
            getGui(){{
                return this.eGui;
            }}
        }}
        """
    )

@lru_cache(maxsize=1)
def _reflect_tickets_table() -> Table:
    engine = get_tickets_engine()
    metadata = MetaData()
    try:
        table = Table("tickets", metadata, autoload_with=engine)
    except NoSuchTableError as exc:
        logger.error("Tickets table not found in the configured database.")
        raise RuntimeError(
            "Tickets table not found in the configured database. Please create it before running the app."
        ) from exc
    return table


def _ticket_mutable_columns() -> list[str]:
    return [col.name for col in _reflect_tickets_table().columns if col.name != "id"]


@lru_cache(maxsize=1)
def _reflect_ticket_comments_table() -> Table:
    engine = get_tickets_engine()
    metadata = MetaData()
    try:
        table = Table("ticket_comments", metadata, autoload_with=engine)
    except NoSuchTableError as exc:
        logger.error("ticket_comments table not found in the configured database.")
        raise RuntimeError(
            "ticket_comments table not found in the configured database. Please create it before using ticket comments."
        ) from exc
    return table


def _fetch_ticket_comments(ticket_id: int, *, engine: Optional[Engine] = None) -> list[dict[str, Any]]:
    engine = engine or get_tickets_engine()
    table = _reflect_ticket_comments_table()
    stmt = (
        select(table)
        .where(table.c.ticket_id == ticket_id)
        .order_by(table.c.created_at.asc(), table.c.id.asc())
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt)
        return [dict(row._mapping) for row in rows]


def _fetch_ticket_comment_counts(*, engine: Optional[Engine] = None) -> dict[int, int]:
    engine = engine or get_tickets_engine()
    try:
        table = _reflect_ticket_comments_table()
    except RuntimeError:
        return {}

    stmt = select(table.c.ticket_id, func.count().label("count")).group_by(table.c.ticket_id)
    try:
        with engine.connect() as conn:
            result = conn.execute(stmt)
            counts: dict[int, int] = {}
            for ticket_id, count in result:
                if ticket_id is None:
                    continue
                try:
                    counts[int(ticket_id)] = int(count or 0)
                except (TypeError, ValueError):
                    continue
            return counts
    except SQLAlchemyError as exc:
        logger.exception("Failed to aggregate ticket comment counts: %s", exc)
        return {}


def _create_ticket_comment(
    ticket_id: int,
    author: str,
    body: str,
    *,
    engine: Optional[Engine] = None,
) -> dict[str, Any]:
    cleaned_author = (author or "").strip()
    cleaned_body = (body or "").strip()
    if not cleaned_author:
        raise ValueError("Author name is required to save a comment.")
    if not cleaned_body:
        raise ValueError("Comment text cannot be empty.")

    engine = engine or get_tickets_engine()
    table = _reflect_ticket_comments_table()
    payload = {
        "ticket_id": ticket_id,
        "author": cleaned_author,
        "body": cleaned_body,
    }
    if "created_at" in table.c:
        payload.setdefault("created_at", _now_local())
    stmt = insert(table).values(payload).returning(table)
    with engine.begin() as conn:
        result = conn.execute(stmt)
        inserted = result.mappings().first()
    return dict(inserted) if inserted is not None else payload


def _handle_ticket_comment_submit(
    ticket_id: int,
    *,
    body_state_key: str,
    comment_success_key: str,
    comment_error_key: str,
) -> None:
    body_value = (st.session_state.get(body_state_key) or "").strip()
    if not body_value:
        st.session_state[comment_error_key] = "Comment cannot be empty."
        st.session_state.pop(comment_success_key, None)
        return

    try:
        _create_ticket_comment(ticket_id, CURRENT_USER_NAME, body_value)
    except ValueError as exc:
        st.session_state[comment_error_key] = str(exc)
        st.session_state.pop(comment_success_key, None)
    except Exception as exc:
        logger.exception("Failed to save comment for ticket %s: %s", ticket_id, exc)
        st.session_state[comment_error_key] = "Couldn't save your comment. Please try again."
        st.session_state.pop(comment_success_key, None)
    else:
        st.session_state[comment_success_key] = "Comment posted."
        st.session_state.pop(comment_error_key, None)
        st.session_state[body_state_key] = ""


def _format_comment_timestamp(value: Any) -> str:
    if value is None:
        return ""

    dt_value: Optional[datetime]
    if isinstance(value, datetime):
        dt_value = _to_local_naive(value)
    else:
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            parsed = pd.NaT
        if pd.isna(parsed):
            return ""
        if isinstance(parsed, pd.Timestamp):
            parsed = parsed.to_pydatetime()
        if isinstance(parsed, datetime):
            dt_value = _to_local_naive(parsed)
        elif isinstance(parsed, date):
            dt_value = datetime.combine(parsed, time.min)
        else:
            return str(value)

    if dt_value is None:
        return ""

    formatted = dt_value.strftime("%b %d, %Y · %I:%M %p")
    return formatted.replace(" 0", " ")


def _extract_database_url_from_mapping(data: Mapping[str, Any]) -> Optional[str]:
    for key in DATABASE_URL_ENV_KEYS:
        raw = data.get(key)
        if raw is not None:
            text = str(raw).strip()
            if text:
                return text

    for value in data.values():
        if isinstance(value, Mapping):
            nested = _extract_database_url_from_mapping(value)
            if nested:
                return nested
    return None


def _load_local_database_url() -> Optional[str]:
    if not LOCAL_SECRETS_PATH.exists():
        logger.debug("Local secrets file %s not found.", LOCAL_SECRETS_PATH)
        return None
    try:
        logger.info("Attempting to load database URL from local secrets file %s.", LOCAL_SECRETS_PATH)
        with LOCAL_SECRETS_PATH.open("rb") as handle:
            contents = tomllib.load(handle)
    except Exception as exc:  # pragma: no cover - startup failure surface
        logger.error("Failed to read %s: %s", LOCAL_SECRETS_FILENAME, exc)
        raise RuntimeError(
            f"Failed to parse database credentials from {LOCAL_SECRETS_FILENAME}: {exc}"
        ) from exc

    if not isinstance(contents, Mapping):
        logger.error("%s does not contain a TOML mapping.", LOCAL_SECRETS_FILENAME)
        raise RuntimeError(
            f"{LOCAL_SECRETS_FILENAME} must contain TOML key/value pairs."
        )

    url = _extract_database_url_from_mapping(contents)
    if not url:
        logger.error(
            "%s is missing required database URL keys (%s).",
            LOCAL_SECRETS_FILENAME,
            ", ".join(DATABASE_URL_ENV_KEYS),
        )
        raise RuntimeError(
            f"{LOCAL_SECRETS_FILENAME} exists but is missing a database URL. "
            f"Set one of {', '.join(DATABASE_URL_ENV_KEYS)} in that file."
        )
    logger.info("Loaded database URL from local secrets file.")
    return url


def _load_secrets_database_url() -> Optional[str]:
    try:
        secrets_dict: Dict[str, Any] = dict(st.secrets)
    except Exception:
        secrets_dict = {}
    if not secrets_dict:
        logger.debug("Streamlit secrets not available or empty.")
        return None
    logger.info("Attempting to load database URL from Streamlit secrets.")
    return _extract_database_url_from_mapping(secrets_dict)


def _normalize_database_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Database URL is empty.")
    lowered = raw.lower()
    mappings = {
        "postgres://": "postgresql+psycopg://",
        "postgresql://": "postgresql+psycopg://",
        "postgresql+asyncpg://": "postgresql+psycopg://",
    }
    for legacy, target in mappings.items():
        if lowered.startswith(legacy):
            raw = target + raw[len(legacy):]
            break
    if "sslmode=" not in raw.lower():
        raw = f"{raw}{'&' if '?' in raw else '?'}sslmode=require"
    return raw


def _get_database_url() -> str:
    cache_key = "_tickets_database_url"
    try:
        cached = st.session_state.get(cache_key)
    except Exception:
        cached = None
    if cached:
        logger.debug("Using cached database URL from session state.")
        return cached

    local_url = _load_local_database_url()
    if local_url:
        normalized = _normalize_database_url(local_url)
        try:
            st.session_state[cache_key] = normalized
        except Exception:
            pass
        logger.info("Database URL resolved from local secrets file.")
        return normalized

    secrets_url = _load_secrets_database_url()
    if secrets_url:
        normalized = _normalize_database_url(secrets_url)
        try:
            st.session_state[cache_key] = normalized
        except Exception:
            pass
        logger.info("Database URL resolved from Streamlit secrets.")
        return normalized

    for key in DATABASE_URL_ENV_KEYS:
        env_val = os.getenv(key)
        if env_val:
            normalized = _normalize_database_url(env_val)
            try:
                st.session_state[cache_key] = normalized
            except Exception:
                pass
            logger.info("Database URL resolved from environment variable '%s'.", key)
            return normalized

    logger.error(
        "Database URL not configured. Looked in local secrets, Streamlit secrets, and environment variables (%s).",
        ", ".join(DATABASE_URL_ENV_KEYS),
    )
    raise RuntimeError(
        "Database URL not configured. Provide credentials in "
        f"{LOCAL_SECRETS_FILENAME}, Streamlit secrets, or environment variables ({', '.join(DATABASE_URL_ENV_KEYS)})."
    )


@st.cache_resource(show_spinner=False)
def get_tickets_engine() -> Engine:
    url = _get_database_url()
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600, future=True)

def _serialize_field(column: str, value: Any):
    if column in BOOLEAN_COLUMNS:
        return bool(value)

    if value is None or (isinstance(value, str) and not value.strip()):
        return None

    if column in DATE_COLUMNS:
        return _coerce_date(value)

    if isinstance(value, bool):
        return value

    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        return datetime.combine(value, time.min)

    if isinstance(value, str):
        return value

    if isinstance(value, time):
        return value.replace(microsecond=0).isoformat()

    if isinstance(value, (int, float)):
        return value

    return str(value)


def _coerce_date(value: Any) -> Optional[date]:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):  # excludes datetime thanks to check above
        return value

    if isinstance(value, pd.Timestamp):
        return value.date()

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return date.fromisoformat(text)
        except ValueError:
            try:
                return datetime.fromisoformat(text).date()
            except ValueError:
                try:
                    parsed = pd.to_datetime(text, errors="raise")
                except Exception:
                    logging.warning("Could not parse date value '%s'", text)
                    return None
                if pd.isna(parsed):
                    return None
                return parsed.date()

    logging.warning("Unsupported date value type %s (%s)", type(value), value)
    return None


def _sanitize_ticket_payload_for_insert(data: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for column in _ticket_mutable_columns():
        if column == "created_at":
            value = data.get(column)
            if isinstance(value, datetime):
                payload[column] = _to_local_naive(value)
            elif isinstance(value, date):
                payload[column] = datetime.combine(value, time.min)
            else:
                payload[column] = _now_local()
            continue
        payload[column] = _serialize_field(column, data.get(column))

    if not payload.get("ticket_group"):
        raise ValueError("ticket_group is required to create a ticket.")
    if not payload.get("created_by"):
        payload["created_by"] = None
    return payload


def _sanitize_ticket_update(patch: Dict[str, Any]) -> Dict[str, Any]:
    allowed = set(_reflect_tickets_table().c.keys()) - {"id", "created_at"}
    sanitized: Dict[str, Any] = {}
    for key, value in patch.items():
        if key not in allowed:
            continue
        sanitized[key] = _serialize_field(key, value)
    return sanitized


def _format_datetime_for_display(value: Any) -> str:
    if isinstance(value, datetime):
        local_dt = _to_local_naive(value)
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, str):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min).strftime("%Y-%m-%d %H:%M:%S")
    return ""


def _fetch_tickets_dataframe(engine: Engine) -> pd.DataFrame:
    table = _reflect_tickets_table()
    with engine.connect() as conn:
        result = conn.execute(select(table).order_by(table.c.created_at.desc(), table.c.id.desc()))
        rows = [dict(row._mapping) for row in result]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=TICKET_COLUMN_ORDER)

    for col in TICKET_COLUMN_ORDER:
        if col not in df.columns:
            df[col] = False if col in BOOLEAN_COLUMNS else ""

    if "reminder_enabled" in df.columns:
        df["reminder_enabled"] = df["reminder_enabled"].fillna(False).astype(bool)

    if "created_at" in df.columns:
        df["created_at"] = df["created_at"].apply(_format_datetime_for_display)

    for col in set(TICKET_COLUMN_ORDER) - BOOLEAN_COLUMNS - {"created_at"}:
        if col in df.columns:
            df[col] = df[col].fillna("")

    comment_counts: dict[int, int] = {}
    try:
        comment_counts = _fetch_ticket_comment_counts(engine=engine)
    except Exception as exc:
        logger.exception("Failed to fetch ticket comment counts: %s", exc)
        comment_counts = {}

    if "id" in df.columns and not df.empty:
        id_numeric = pd.to_numeric(df["id"], errors="coerce")
        df["comment_count"] = [
            int(comment_counts.get(int(val), 0)) if pd.notna(val) else 0
            for val in id_numeric
        ]
    else:
        df["comment_count"] = 0

    try:
        df["comment_count"] = df["comment_count"].fillna(0).astype(int)
    except Exception:
        df["comment_count"] = df["comment_count"].fillna(0)

    ordered_cols = [c for c in TICKET_COLUMN_ORDER if c in df.columns]
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    if ordered_cols:
        df = df[ordered_cols + extra_cols]
    return df


def _update_ticket_sequence(df: pd.DataFrame) -> None:
    if "id" in df.columns and not df.empty:
        try:
            seq_base = pd.to_numeric(df["id"], errors="coerce").max()
            st.session_state.ticket_seq = int(seq_base) + 1 if pd.notna(seq_base) else 1
        except Exception:
            st.session_state.ticket_seq = 1
    else:
        st.session_state.ticket_seq = 1


def _refresh_tickets_df(engine: Optional[Engine] = None) -> None:
    try:
        engine = engine or get_tickets_engine()
        df = _fetch_tickets_dataframe(engine)
    except Exception as exc:  # pragma: no cover - Streamlit runtime logging
        logging.exception("Failed to refresh tickets from database: %s", exc)
        st.error("Failed to refresh tickets from the database. Please verify the connection and reload the app.")
        raise
    st.session_state.tickets_df = df
    _update_ticket_sequence(df)


def init_ticket_store() -> None:
    if st.session_state.get("_tickets_initialized"):
        return

    engine = get_tickets_engine()
    try:
        df = _fetch_tickets_dataframe(engine)
    except Exception as exc:  # pragma: no cover - Streamlit runtime logging
        logging.exception("Database unavailable: %s", exc)
        st.error("Database unavailable. Please verify the connection and refresh the app.")
        raise

    st.session_state.tickets_df = df
    _update_ticket_sequence(df)

    st.session_state["_tickets_initialized"] = True


# --- Helper: color missing field labels in red (post-submit) ---
def _color_missing_labels(label_texts):
    if not label_texts:
        return
    sels = []
    for lab in label_texts:
        sels.append(f'label:has(+ div input[aria-label="{lab}"])')
        sels.append(f'label:has(+ div textarea[aria-label="{lab}"])')
        sels.append(f'label:has(+ div [role="combobox"][aria-label="{lab}"])')
        sels.extend(REMINDER_LABEL_SELECTORS.get(lab, []))
    css = "<style>" + ", ".join(sels) + "{color:#dc2626 !important;font-weight:600!important;}</style>"
    st.markdown(css, unsafe_allow_html=True)

# Top-level tab icons
TAB_ICONS = {
    "Home": "home",
    "Digicare Tickets": "report_problem",
    "Call Tickets": "phone",
    "Requests": "article",
    "Card": "credit_score",
    "Client": "group",
    "CPE": "router",
    "IVR": "support_agent",
    "Settings": "settings",
    "Dahi Nemutlu": "notifications",
    "Exit": "exit_to_app",
}
TAB_NAMES = list(TAB_ICONS.keys())


def _collect_due_reminders() -> list[dict[str, Any]]:
    df = st.session_state.get("tickets_df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    required = {"reminder_enabled", "reminder_at"}
    if not required.issubset(df.columns):
        return []

    enabled_series = df["reminder_enabled"]
    if enabled_series.dtype == bool:
        enabled_mask = enabled_series
    else:
        enabled_mask = (
            enabled_series.astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y", "t", "on"})
        )

    reminders_df = df.loc[enabled_mask].copy()
    if reminders_df.empty:
        return []

    reminder_times = pd.to_datetime(reminders_df["reminder_at"], errors="coerce")
    if isinstance(getattr(reminder_times, "dtype", None), pd.DatetimeTZDtype):
        reminder_times = reminder_times.dt.tz_convert(APP_TIMEZONE).dt.tz_localize(None)

    now = _now_local()
    due_mask = reminder_times.notna() & (reminder_times <= now)

    filtered = reminders_df.loc[due_mask].copy()
    if filtered.empty:
        return []

    filtered["__reminder_at"] = reminder_times.loc[filtered.index]
    filtered = filtered.sort_values("__reminder_at")

    def _clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        return text

    def _normalize_identifier(raw: Any) -> str:
        text = _clean(raw)
        if not text:
            return ""
        try:
            num = float(text)
            if num.is_integer():
                return str(int(num))
        except ValueError:
            pass
        return text

    fallback_ticket_type_cols = {
        "activities & inquiries": "activity_inquiry_type",
        "complaints": "complaint_type",
        "osp appointments": "osp_type",
    }

    generic_ticket_type_cols = ("ticket_type", "activity_inquiry_type", "complaint_type", "osp_type")

    reminders: list[dict[str, Any]] = []
    for _, row in filtered.iterrows():
        dt_value = row.get("__reminder_at")
        if pd.isna(dt_value):
            continue
        scheduled_for = dt_value.to_pydatetime() if hasattr(dt_value, "to_pydatetime") else dt_value
        if isinstance(scheduled_for, datetime):
            scheduled_for = _to_local_naive(scheduled_for)

        ticket_group_value = _clean(row.get("ticket_group"))

        ticket_type_value = _clean(row.get("ticket_type"))
        if not ticket_type_value and ticket_group_value:
            fallback_col = fallback_ticket_type_cols.get(ticket_group_value.lower())
            if fallback_col:
                ticket_type_value = _clean(row.get(fallback_col))
        if not ticket_type_value:
            for col in generic_ticket_type_cols:
                candidate = _clean(row.get(col))
                if candidate:
                    ticket_type_value = candidate
                    break

        reminders.append(
            {
                "ticket_id": _normalize_identifier(row.get("id")),
                "ont_id": _normalize_identifier(row.get("ont_id")),
                "ticket_group": ticket_group_value,
                "ticket_type": ticket_type_value,
                "complaint_status": _clean(row.get("complaint_status")),
                "note": _clean(row.get("reminder_note")),
                "recipient": _clean(row.get("reminder_recipient")),
                "scheduled_for": scheduled_for,
            }
        )

    return reminders

def _prepare_ticket_row(row: Dict[str, Any]) -> Dict[str, Any]:
    data = row.copy()
    default_creator = globals().get("DEFAULT_CREATED_BY", CURRENT_USER_NAME)
    if not data.get("created_by"):
        assigned = data.pop("assigned_to", None)
        data["created_by"] = assigned or default_creator
    else:
        data.pop("assigned_to", None)
    data.setdefault("reminder_enabled", False)
    data.setdefault("reminder_recipient", "")
    data.setdefault("reminder_note", "")
    data.setdefault("reminder_at", "")
    data.setdefault("channel", "")
    data.setdefault("visit_required", False)
    return data


CURRENT_USER_NAME = "Dahi Nemutlu"

def _is_popup_view() -> bool:
    return str(st.query_params.get("popup", "")).strip().lower() == "call_ticket"


def _schedule_popup_close():
    if _is_popup_view():
        st.session_state["_close_call_ticket_popup"] = True


def _ensure_popup_script():
    if st.session_state.get("_popup_script_ready"):
        return
    components.html(
        """
        <script>
        (function(){
            const root = (window.parent && window.parent !== window) ? window.parent : window;
            if (root.openCallTicketPopup) {
                return;
            }

                        const tryFocus = (win) => {
                if (!win) return;
                try {
                    win.focus();
                } catch (err) {
                    console.warn('Popup focus failed', err);
                }
            };

            const hideExistingModal = () => {
                const modal = root.document.getElementById('call-ticket-existing-popup-modal');
                if (modal) {
                    modal.style.display = 'none';
                    modal.dataset.path = '';
                    modal.dataset.toggleId = '';
                }
            };

            const ensureExistingModal = () => {
                let modal = root.document.getElementById('call-ticket-existing-popup-modal');
                if (modal) {
                    return modal;
                }
                modal = root.document.createElement('div');
                modal.id = 'call-ticket-existing-popup-modal';
                modal.setAttribute('style', [
                    'position:fixed',
                    'inset:0',
                    'background:rgba(15,23,42,0.45)',
                    'display:none',
                    'align-items:center',
                    'justify-content:center',
                    'z-index:10000',
                    'padding:1.5rem'
                ].join(';'));
                modal.innerHTML = `
                    <div class="ct-modal-card">
                        <h3 class="ct-modal-title">Start a new ticket?</h3>
                        <p class="ct-modal-body">You already have a Call Ticket popup open with unsaved changes. Starting a new ticket will discard them. Continue?</p>
                        <div class="ct-modal-actions">
                            <button type="button" data-action="cancel" class="ct-modal-btn ct-modal-btn-secondary">Return</button>
                            <button type="button" data-action="proceed" class="ct-modal-btn ct-modal-btn-secondary">Start new ticket</button>
                        </div>
                    </div>
                `;
                modal.addEventListener('click', (event) => {
                    if (event.target === modal) {
                        hideExistingModal();
                        const existing = root.__call_ticket_popup_ref;
                        if (existing && !existing.closed) {
                            tryFocus(existing);
                        }
                    }
                });
                const cancelBtn = modal.querySelector('[data-action="cancel"]');
                const proceedBtn = modal.querySelector('[data-action="proceed"]');
                cancelBtn.addEventListener('click', () => {
                    hideExistingModal();
                    const existing = root.__call_ticket_popup_ref;
                    if (existing && !existing.closed) {
                        tryFocus(existing);
                    }
                });
                proceedBtn.addEventListener('click', () => {
                    const path = modal.dataset.path || '';
                    const toggleId = modal.dataset.toggleId || '';
                    hideExistingModal();
                    root.openCallTicketPopup(path, toggleId || null, { confirmIfOpen: false, _fromModal: true });
                });
                root.document.body.appendChild(modal);
                if (!root.__call_ticket_existing_modal_keydown_bound) {
                    root.__call_ticket_existing_modal_keydown_bound = true;
                    root.addEventListener('keydown', (event) => {
                        if (event.key === 'Escape') {
                            const currentModal = root.document.getElementById('call-ticket-existing-popup-modal');
                            if (currentModal && currentModal.style.display && currentModal.style.display !== 'none') {
                                hideExistingModal();
                                const existing = root.__call_ticket_popup_ref;
                                if (existing && !existing.closed) {
                                    tryFocus(existing);
                                }
                            }
                        }
                    });
                }
                return modal;
            };

            const showExistingModal = (payload) => {
                const modal = ensureExistingModal();
                modal.dataset.path = payload.path || '';
                modal.dataset.toggleId = payload.toggleId || '';
                modal.style.display = 'flex';
            };

            root.openCallTicketPopup = function(path, toggleId, opts) {
                const options = opts || {};
                const confirmIfOpen = !!options.confirmIfOpen && !options._fromModal;
                const popupName = 'call_ticket_native_popup';
                const makeUrl = (input) => {
                    if (!input) return root.location.href;
                    if (/^https?:/i.test(input)) return input;
                    if (input.startsWith('?')) {
                        return root.location.origin + root.location.pathname + input;
                    }
                    if (input.startsWith('/')) {
                        return root.location.origin + input;
                    }
                    return input;
                };

                if (confirmIfOpen) {
                    const existing = root.__call_ticket_popup_ref;
                    if (existing && !existing.closed) {
                        if (toggleId) {
                            const toggle = root.document.getElementById(toggleId);
                            if (toggle) {
                                toggle.checked = false;
                            }
                        }
                        showExistingModal({ path, toggleId });
                        tryFocus(existing);
                        return false;
                    }
                }

                const fullUrl = makeUrl(path);
                try {
                    const features = [
                        'popup=yes',
                        'width=1280',
                        'height=768',
                        'left=160',
                        'top=120',
                        'location=no',
                        'menubar=no',
                        'toolbar=no',
                        'status=no',
                        'resizable=yes',
                        'scrollbars=yes'
                    ].join(',');
                    const popup = root.open(fullUrl, popupName, features);
                    if (popup && !popup.closed) {
                        root.__call_ticket_popup_ref = popup;
                        popup.onbeforeunload = function() {
                            try {
                                root.__call_ticket_popup_ref = null;
                            } catch (cleanupErr) {
                                console.warn('Popup cleanup failed', cleanupErr);
                            }
                        };
                    }
                    if (toggleId) {
                        const toggle = root.document.getElementById(toggleId);
                        if (toggle) {
                            toggle.checked = false;
                        }
                    }
                    if (popup && !popup.closed) {
                        tryFocus(popup);
                    } else {
                        root.location.href = fullUrl;
                    }
                } catch (err) {
                    root.location.href = fullUrl;
                }
                return false;
            };
            const ensureHandler = () => {
                if (root.__call_ticket_popup_handler_ready) {
                    return;
                }
                root.__call_ticket_popup_handler_ready = true;
                const handler = (event) => {
                    const target = event.target && event.target.closest ? event.target.closest('[data-call-ticket-popup]') : null;
                    if (!target) {
                        return;
                    }
                    event.preventDefault();
                    event.stopPropagation();
                    const pathRaw = target.getAttribute('data-popup-href') || target.getAttribute('href');
                    const path = pathRaw ? pathRaw.replace(/&amp;/g, '&') : pathRaw;
                    const toggleId = target.getAttribute('data-toggle-id');
                    const confirmAttr = target.getAttribute('data-confirm-existing-popup');
                    const confirmIfOpen = confirmAttr === '1' || confirmAttr === 'true';
                    root.openCallTicketPopup(path, toggleId, { confirmIfOpen });
                };
                root.document.addEventListener('click', handler, true);
            };
            if (root.document.readyState === 'loading') {
                root.document.addEventListener('DOMContentLoaded', ensureHandler, { once: true });
            } else {
                ensureHandler();
            }
        })();
        </script>
        """,
        height=0,
    )
    st.session_state["_popup_script_ready"] = True


def _ensure_popup_bridge():
    if st.session_state.get("_popup_bridge_ready"):
        return
    components.html(
        """
        <script>
        (function(){
            const root = (window.parent && window.parent !== window) ? window.parent : window;
            if (root.__call_ticket_popup_bridge_ready) {
                return;
            }
            root.__call_ticket_popup_bridge_ready = true;
            const script = root.document.createElement('script');
            script.type = 'text/javascript';
            script.text = `(() => {
                const popupName = 'call_ticket_native_popup';
                const features = [
                    'popup=yes',
                    'width=1280',
                    'height=768',
                    'left=160',
                    'top=120',
                    'location=no',
                    'menubar=no',
                    'toolbar=no',
                    'status=no',
                    'resizable=yes',
                    'scrollbars=yes'
                ].join(',');

                const tryFocus = (win) => {
                    if (!win) return;
                    try {
                        win.focus();
                    } catch (err) {
                        console.warn('Popup focus failed', err);
                    }
                };

                const registerWithOpener = () => {
                    if (!window.opener) return;
                    try {
                        window.opener.__call_ticket_popup_ref = window;
                    } catch (err) {
                        console.warn('Popup parent sync failed', err);
                    }
                };

                registerWithOpener();
                window.addEventListener('beforeunload', () => {
                    if (!window.opener) return;
                    try {
                        window.opener.__call_ticket_popup_ref = null;
                    } catch (err) {
                        console.warn('Popup parent cleanup failed', err);
                    }
                });

                if (window.name === popupName) {
                    tryFocus(window);
                    return;
                }

                if (!window.opener) {
                    try {
                        const popup = window.open(window.location.href, popupName, features);
                        if (popup && !popup.closed) {
                            window.name = popupName + '_bridge';
                            const msg = document.createElement('div');
                            msg.setAttribute('style', 'font-family:Inter, sans-serif;padding:2rem;text-align:center;color:#0f172a;');
                            msg.textContent = 'Launching Call Ticket window…';
                            document.body.innerHTML = '';
                            document.body.appendChild(msg);
                            setTimeout(() => {
                                tryFocus(popup);
                                window.close();
                            }, 500);
                        }
                    } catch (err) {
                        console.warn('Popup bridge failed', err);
                    }
                } else {
                    window.name = popupName;
                    tryFocus(window);
                }
            })();`;
            root.document.head.appendChild(script);
        })();
        </script>
        """,
        height=0,
    )
    st.session_state["_popup_bridge_ready"] = True


def _coerce_ticket_identifier(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if "." in text:
            return int(float(text))
        return int(text)
    except (TypeError, ValueError):
        return None


def _activate_ticket_detail(ticket_id: Any) -> Optional[int]:
    """Normalize and store the ticket id so the detail page can render."""
    normalized = _coerce_ticket_identifier(ticket_id)
    if normalized is None:
        return None
    st.session_state["_ticket_detail_id"] = normalized
    st.session_state["_show_ticket_detail"] = True
    st.session_state["_ticket_detail_reset_flag"] = True
    st.session_state.active_tab = "Call Tickets"
    st.session_state.active_subtab = "Tickets"
    return normalized


def _clear_ticket_detail_view() -> None:
    """Reset state and remove any ticket_detail query parameter."""
    current_id = st.session_state.get("_ticket_detail_current_id")
    _reset_ticket_detail_widgets(current_id)
    st.session_state.pop("_ticket_detail_id", None)
    st.session_state.pop("_show_ticket_detail", None)
    st.session_state.pop("_last_detail_token", None)
    st.session_state.pop("_ticket_detail_current_id", None)
    st.session_state.pop("_ticket_detail_original", None)
    st.session_state.pop("_ticket_detail_message", None)
    st.session_state.pop("_ticket_detail_error", None)
    st.session_state.pop("_ticket_detail_reset_flag", None)
    try:
        if "ticket_detail" in st.query_params:
            del st.query_params["ticket_detail"]
    except Exception:
        pass


def _reset_ticket_detail_widgets(ticket_id: Any) -> None:
    if ticket_id is None:
        return
    prefix = f"ticket_detail_{ticket_id}_"
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(prefix):
            try:
                st.session_state.pop(key)
            except Exception:
                pass


@lru_cache()
def _get_brand_logo_data_uri() -> Optional[str]:
    logo_path = Path(__file__).with_name("assets").joinpath("fibercare.png")
    try:
        with logo_path.open("rb") as fh:
            encoded = b64encode(fh.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def _render_ticket_detail_page(ticket_id: int) -> None:
    """Render a dedicated page showing all stored columns for a ticket."""

    def _stringify_detail_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().isoformat(sep=" ", timespec="seconds")
        if isinstance(value, datetime):
            return value.isoformat(sep=" ", timespec="seconds")
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, time):
            return value.isoformat()
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                return str(value)
        return str(value)

    def _coerce_bool_value(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "on"}
        if isinstance(value, (int, float)):
            return bool(value)
        return bool(value)

    def _parse_date_value(value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        try:
            parsed = pd.to_datetime(value)
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        if isinstance(parsed, pd.Timestamp):
            parsed = parsed.to_pydatetime()
        if isinstance(parsed, datetime):
            return parsed.date()
        if isinstance(parsed, date):
            return parsed
        return None

    select_field_config: dict[str, dict[str, Any]] = {
        "ticket_group": {"options": ["Activities & Inquiries", "Complaints", "OSP Appointments"], "allow_blank": False},
        "activity_inquiry_type": {"options": lambda: ACTIVITY_TYPES},
        "call_type": {"options": lambda: CALL_TYPES},
        "digicare_issue_type": {"options": lambda: DIGICARE_ISSUES},
        "complaint_type": {"options": lambda: COMPLAINT_TYPES},
        "refund_type": {"options": lambda: REFUND_TYPES},
        "employee_suggestion": {"options": lambda: EMP_SUGGESTION},
        "device_location": {"options": lambda: DEVICE_LOC},
        "root_cause": {"options": lambda: ROOT_CAUSE},
        "ont_model": {"options": lambda: ONT_MODELS},
        "complaint_status": {"options": lambda: COMP_STATUS},
        "kurdtel_service_status": {"options": lambda: KURDTEL_SERVICE_STATUS},
        "osp_type": {"options": lambda: OSP_TYPES},
        "issue_type": {"options": lambda: ISSUE_TYPES},
        "fttg": {"options": lambda: FTTG_OPTIONS},
        "city": {"options": lambda: CITY_OPTIONS},
        "fttx_job_status": {"options": lambda: FTTX_JOB_STATUS},
        "callback_status": {"options": lambda: CALLBACK_STATUS},
        "callback_reason": {"options": lambda: CALLBACK_REASON},
        "followup_status": {"options": lambda: FOLLOWUP_STATUS},
        "online_game": {"options": lambda: ONLINE_GAMES, "include_other": True},
        "created_by": {"options": lambda: CREATED_BY_OPTIONS or ([DEFAULT_CREATED_BY] if DEFAULT_CREATED_BY else []), "allow_blank": False},
        "channel": {"options": lambda: CHANNEL_OPTIONS, "allow_blank": False},
    }

    def _resolve_select_options(field: str) -> tuple[list[str], bool]:
        meta = select_field_config.get(field)
        if not meta:
            return [], True
        src = meta.get("options", [])
        try:
            raw = src() if callable(src) else src
        except Exception:
            raw = []
        allow_blank = bool(meta.get("allow_blank", True))
        include_other = bool(meta.get("include_other", False))
        options: list[str] = []
        seen: set[str] = set()
        for opt in raw or []:
            if opt is None:
                continue
            text = str(opt).strip()
            if not text:
                continue
            if text not in seen:
                options.append(text)
                seen.add(text)
        if include_other and "Other" not in seen:
            options.append("Other")
            seen.add("Other")
        return options, allow_blank

    idx, row = _get_row_by_id(ticket_id)
    if row is None:
        st.error("Ticket not found or may have been removed.")
        return

    reset_flag = bool(st.session_state.pop("_ticket_detail_reset_flag", False))
    current_id = st.session_state.get("_ticket_detail_current_id")
    if reset_flag or current_id != ticket_id:
        if current_id is not None and current_id != ticket_id:
            _reset_ticket_detail_widgets(current_id)
        _reset_ticket_detail_widgets(ticket_id)
        st.session_state["_ticket_detail_current_id"] = ticket_id
        st.session_state["_ticket_detail_original"] = row.copy()

    original_row = st.session_state.get("_ticket_detail_original")
    if not isinstance(original_row, dict):
        original_row = row.copy()
        st.session_state["_ticket_detail_original"] = original_row

    success_msg = st.session_state.pop("_ticket_detail_message", None)
    error_msg = st.session_state.pop("_ticket_detail_error", None)

    st.markdown(f"## Ticket #{row.get('id', ticket_id)}")

    left_col, right_col = st.columns([2, 3], gap="large")

    df_source = st.session_state.get("tickets_df")
    base_fields = list(df_source.columns) if isinstance(df_source, pd.DataFrame) else list(row.keys())
    seen_fields: set[str] = set()
    field_order: list[str] = []
    for field in base_fields:
        if field not in row or field in seen_fields:
            continue
        field_order.append(field)
        seen_fields.add(field)
    for field in row.keys():
        if field not in seen_fields:
            field_order.append(field)
            seen_fields.add(field)

    form_key = f"ticket_detail_form_{ticket_id}"
    field_keys: list[tuple[str, str]] = []
    with left_col:
        form_placeholder = st.empty()
        with form_placeholder.form(form_key):
            for field in field_order:
                label = field.replace("_", " ").title()
                widget_key = f"ticket_detail_{ticket_id}_{field}"
                field_disabled = field in READ_ONLY_DETAIL_FIELDS
                if field in BOOLEAN_COLUMNS:
                    default_bool = _coerce_bool_value(original_row.get(field))
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = default_bool
                    st.checkbox(label, key=widget_key, disabled=field_disabled)
                elif field == "reminder_recipient":
                    raw_default = original_row.get(field)
                    default_list = _normalize_reminder_selection(_parse_reminder_recipients(raw_default))
                    options = _ensure_recipient_options(REMINDER_RECIPIENT_OPTIONS, default_list)
                    current_value = st.session_state.get(widget_key, default_list)
                    if not isinstance(current_value, list):
                        current_value = _normalize_reminder_selection(_parse_reminder_recipients(current_value))
                    current_value = _normalize_reminder_selection(current_value)
                    st.multiselect(
                        label,
                        options,
                        default=current_value,
                        key=widget_key,
                        disabled=field_disabled,
                    )
                else:
                    select_options, allow_blank = _resolve_select_options(field)
                    if field in select_field_config:
                        options = select_options.copy()
                        if allow_blank and "" not in options:
                            options.insert(0, "")
                        if not options and allow_blank:
                            options = [""]
                        default_text = _stringify_detail_value(original_row.get(field))
                        if default_text and default_text not in options:
                            options.append(default_text)
                        if widget_key not in st.session_state or st.session_state[widget_key] not in options:
                            if default_text in options:
                                st.session_state[widget_key] = default_text
                            elif options:
                                st.session_state[widget_key] = options[0]
                            else:
                                st.session_state[widget_key] = default_text
                        if options:
                            if st.session_state[widget_key] not in options:
                                st.session_state[widget_key] = options[0] if options else ""
                            st.selectbox(
                                label,
                                options,
                                key=widget_key,
                                disabled=field_disabled,
                            )
                        else:
                            st.text_input(label, key=widget_key, disabled=field_disabled)
                    elif field in DATE_COLUMNS:
                        default_date = _parse_date_value(original_row.get(field))
                        if widget_key not in st.session_state:
                            st.session_state[widget_key] = default_date
                        current_value = st.session_state.get(widget_key)
                        if isinstance(current_value, str):
                            current_value = _parse_date_value(current_value)
                            st.session_state[widget_key] = current_value
                        date_value = current_value or default_date or date.today()
                        picker_key = f"{widget_key}_picker"
                        st.session_state.setdefault(picker_key, date_value)
                        selected_date = st.date_input(
                            label,
                            key=picker_key,
                            disabled=field_disabled,
                        )
                        if not field_disabled:
                            st.session_state[widget_key] = selected_date
                        else:
                            st.session_state[widget_key] = current_value
                    else:
                        default_text = _stringify_detail_value(original_row.get(field))
                        if widget_key not in st.session_state:
                            st.session_state[widget_key] = default_text
                        if field in DETAIL_MULTILINE_FIELDS:
                            st.text_area(label, key=widget_key, disabled=field_disabled, height=140)
                        else:
                            st.text_input(label, key=widget_key, disabled=field_disabled)
                field_keys.append((field, widget_key))
            save_clicked = st.form_submit_button("Save changes")

        left_feedback = st.empty()

    if save_clicked:
        patch: Dict[str, Any] = {}
        for field, widget_key in field_keys:
            if field in READ_ONLY_DETAIL_FIELDS:
                continue
            if field in BOOLEAN_COLUMNS:
                new_bool = _coerce_bool_value(st.session_state.get(widget_key))
                orig_bool = _coerce_bool_value(original_row.get(field))
                if new_bool != orig_bool:
                    patch[field] = new_bool
            elif field == "reminder_recipient":
                new_selection = st.session_state.get(widget_key, [])
                if not isinstance(new_selection, list):
                    new_selection = _normalize_reminder_selection(_parse_reminder_recipients(new_selection))
                else:
                    new_selection = _normalize_reminder_selection(new_selection)
                serialized_new = _serialize_reminder_recipients(new_selection)
                orig_serialized = _serialize_reminder_recipients(
                    _normalize_reminder_selection(_parse_reminder_recipients(original_row.get(field)))
                )
                if serialized_new != orig_serialized:
                    patch[field] = serialized_new
            else:
                new_val = st.session_state.get(widget_key)
                new_text = "" if new_val is None else str(new_val)
                orig_text = _stringify_detail_value(original_row.get(field))
                if new_text != orig_text:
                    patch[field] = new_val

        if patch:
            try:
                _update_row(idx, patch)
                st.session_state["_ticket_detail_message"] = "Ticket updated successfully."
            except Exception:
                st.session_state["_ticket_detail_error"] = "Failed to update ticket. Please try again."
        else:
            st.session_state["_ticket_detail_message"] = "No changes to save."

        st.session_state["_ticket_detail_reset_flag"] = True
        st.rerun()

    with right_col:
        if success_msg and left_feedback:
            left_feedback.success(success_msg)
        elif error_msg and left_feedback:
            left_feedback.error(error_msg)
        else:
            if success_msg:
                st.success(success_msg)
            if error_msg:
                st.error(error_msg)

        comment_success_key = f"_ticket_comment_success_{ticket_id}"
        comment_error_key = f"_ticket_comment_error_{ticket_id}"
        comment_success = st.session_state.get(comment_success_key)
        comment_error = st.session_state.get(comment_error_key)

        comments: list[dict[str, Any]] = []
        comments_error: Optional[str] = None
        try:
            comments = _fetch_ticket_comments(ticket_id)
        except RuntimeError as exc:
            comments_error = str(exc)
        except SQLAlchemyError as exc:
            comments_error = "Couldn't load comments right now. Please try again."
            logger.exception("Failed to load comments for ticket %s: %s", ticket_id, exc)
        except Exception as exc:
            comments_error = "Unexpected error while loading comments."
            logger.exception("Unexpected error while loading comments for ticket %s: %s", ticket_id, exc)

        comment_count = len(comments) if not comments_error else 0
        detail_tabs = st.tabs([f"Comments ({comment_count})", "Ticket History"])

        with detail_tabs[0]:
            if comment_success:
                st.success(comment_success)
                st.session_state.pop(comment_success_key, None)
                st.session_state.pop(comment_error_key, None)
            elif comment_error:
                st.error(comment_error)
                st.session_state.pop(comment_error_key, None)
                st.session_state.pop(comment_success_key, None)

            if comments_error:
                st.warning(comments_error)
            else:
                if comments:
                    comment_cards: list[str] = []
                    for comment in comments:
                        author_text = str(comment.get("author") or "Anonymous")
                        timestamp_text = _format_comment_timestamp(comment.get("created_at"))
                        body_text = (comment.get("body") or "").strip()
                        body_html = escape(body_text).replace("\n", "<br />")
                        card_html = (
                            "<div class=\"ticket-comment\">"
                            f"<div class=\"ticket-comment__meta\"><span class=\"ticket-comment__author\">{escape(author_text)}</span>"
                            f"<span class=\"ticket-comment__time\">{escape(timestamp_text)}</span></div>"
                            f"<div class=\"ticket-comment__body\">{body_html}</div>"
                            "</div>"
                        )
                        comment_cards.append(card_html)
                    st.markdown(
                        '<div class="ticket-comments">' + "".join(comment_cards) + '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No comments yet.")

                st.divider()

                body_state_key = f"ticket_comment_body_{ticket_id}"
                st.session_state.setdefault(body_state_key, "")

                st.markdown("#### Add a comment")
                with st.form(f"ticket_comment_form_{ticket_id}"):
                    st.text_area("Comment", key=body_state_key, height=140)
                    st.form_submit_button(
                        "Post comment",
                        on_click=_handle_ticket_comment_submit,
                        kwargs={
                            "ticket_id": ticket_id,
                            "body_state_key": body_state_key,
                            "comment_success_key": comment_success_key,
                            "comment_error_key": comment_error_key,
                        },
                    )

        with detail_tabs[1]:
            st.write("TBD")

# -------- State / routing --------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Call Tickets"

# Honor query param
if "tab" in st.query_params:
    t = st.query_params["tab"]
    # Backward compatibility for old links
    if t == "Call Center Tickets":
        t = "Call Tickets"
    if t in TAB_NAMES:
        st.session_state.active_tab = t


# Load CSS
with open("app_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_ticket_store()

try:
    _detail_param = st.query_params.get("ticket_detail")
except Exception:
    _detail_param = None
if isinstance(_detail_param, list):
    _detail_param = _detail_param[0] if _detail_param else None
if _detail_param not in (None, "", "None"):
    if _activate_ticket_detail(_detail_param) is None:
        try:
            if "ticket_detail" in st.query_params:
                del st.query_params["ticket_detail"]
        except Exception:
            pass
else:
    if st.session_state.get("_show_ticket_detail"):
        _clear_ticket_detail_view()

if _is_popup_view():
    if st.session_state.get("_close_call_ticket_popup"):
        components.html(
            """
            <script>
            (function(){
                const root = (window.parent && window.parent !== window) ? window.parent : window;
                if (root.__call_ticket_popup_close_dispatched) {
                    return;
                }
                root.__call_ticket_popup_close_dispatched = true;
                const script = root.document.createElement('script');
                script.type = 'text/javascript';
                script.text = `(() => {
                    const finalizeOpener = () => {
                        try {
                            if (window.opener && !window.opener.closed) {
                                try { window.opener.__call_ticket_popup_ref = null; } catch (err) {}
                                try { window.opener.focus(); } catch (err) {}
                            }
                        } catch (err) {}
                    };

                    const tryCloseSelf = () => {
                        finalizeOpener();
                        try { window.close(); } catch (err) {}
                        setTimeout(() => {
                            try { window.close(); } catch (err) {}
                        }, 150);
                    };

                    try {
                        if (window.name === 'call_ticket_native_popup') {
                            tryCloseSelf();
                            return;
                        }
                        if (window.__call_ticket_popup_ref && !window.__call_ticket_popup_ref.closed) {
                            try {
                                window.__call_ticket_popup_ref.close();
                            } catch (err) {}
                            return;
                        }
                        if (window.opener && window.opener.__call_ticket_popup_ref && !window.opener.__call_ticket_popup_ref.closed) {
                            try {
                                window.opener.__call_ticket_popup_ref.close();
                            } catch (err) {}
                            finalizeOpener();
                            return;
                        }
                    } catch (err) {}

                    tryCloseSelf();
                })();`;
                root.document.head.appendChild(script);
            })();
            </script>
            """,
            height=0,
        )
        st.session_state["_close_call_ticket_popup"] = False
        st.stop()

    _ensure_popup_bridge()
    if not st.session_state.get("_popup_layout_bootstrapped"):
        st.session_state.active_tab = "Call Tickets"
        st.session_state.active_subtab = "Tickets"
        st.session_state.show_new_form = True
        if not st.session_state.get("_new_ticket_focus"):
            st.session_state["_new_ticket_focus"] = "Complaints"
        st.session_state["_popup_layout_bootstrapped"] = True

# -------- Top bar --------
if not _is_popup_view():
    reminders = _collect_due_reminders()
    notif_count = len(reminders)

    html = ['<div class="topbar">']
    html.append('<input type="checkbox" id="notifications-toggle" class="notifications-toggle" />')
    logo_uri = _get_brand_logo_data_uri()
    logo_markup = f'<img src="{logo_uri}" alt="FiberCare logo" />' if logo_uri else ""
    html.append(f'<div class="brand">{logo_markup}<span class="brandStack">FiberCare</span></div>')

    # Build tab items once for reuse in inline row and hamburger menu
    tab_items = []
    for name in TAB_NAMES:
        icon = f'<span class="material-icons">{TAB_ICONS[name]}</span>'
        if name == "Call Tickets":
            active_cls = " call-center-active" if st.session_state.active_tab == name else ""
            tab_items.append(
                f'<a href="?tab=Call%20Tickets" target="_self" class="tab{active_cls}">{icon} {name}</a>'
            )
        elif name == "Client":
            active_cls = " call-center-active" if st.session_state.active_tab == name else ""
            tab_items.append(
                f'<a href="?tab=Client" target="_self" class="tab{active_cls}">{icon} {name}</a>'
            )
        elif name == "Dahi Nemutlu":
            badge = f'<span class="notif-pill">{notif_count}</span>' if notif_count else ""
            tab_items.append(
                '<span class="tab notifications-tab">'
                f'<label for="notifications-toggle" class="notifications-trigger" aria-label="Notifications">{icon}{badge}</label>'
                f'<span class="notifications-name">{name}</span>'
                '</span>'
            )
        elif name == "Exit":
            tab_items.append(f'<span class="tab-disabled" title="Exit">{icon}</span>')
        else:
            active_cls = " active" if st.session_state.active_tab == name else ""
            tab_items.append(f'<span class="tab-disabled{active_cls}">{icon} {name}</span>')

    # Inline tabs row
    html.append('<div class="tabs" id="topbar-tabs">')
    html.extend(tab_items)
    html.append('</div>')

    # Burger toggle (CSS only) and overlay drawer menu
    html.append('<input type="checkbox" id="topbar-burger-toggle" class="burger-toggle" />')
    html.append('<label for="topbar-burger-toggle" class="burger" id="topbar-burger" aria-label="Menu"><span class="material-icons">menu</span></label>')
    html.append('<div class="hamburger-overlay" id="topbar-overlay">')
    html.append('<label for="topbar-burger-toggle" class="overlay-backdrop"></label>')
    html.append('<div class="hamburger-drawer">')
    html.append('<div class="menu-header"><div class="title">Menu</div><label for="topbar-burger-toggle" class="close-btn" aria-label="Close"><span class="material-icons">close</span></label></div>')
    html.append('<div class="menu-items">')
    html.extend(tab_items)
    html.append('</div></div></div>')

    # Notifications overlay (desktop & mobile)
    notifications_content: list[str] = []
    now_local = _now_local()
    if reminders:
        notifications_content.append('<div class="notifications-cards">')
        for reminder in reminders:
            scheduled_for = reminder.get("scheduled_for")
            if isinstance(scheduled_for, pd.Timestamp):
                scheduled_dt = scheduled_for.to_pydatetime()
            else:
                scheduled_dt = scheduled_for if isinstance(scheduled_for, datetime) else None
            if scheduled_dt is None:
                parsed_dt = pd.to_datetime(scheduled_for, errors="coerce")
                if pd.isna(parsed_dt):
                    continue
                scheduled_dt = parsed_dt.to_pydatetime()
            if scheduled_dt is not None:
                scheduled_dt = _to_local_naive(scheduled_dt)

            time_display = scheduled_dt.strftime("%I:%M %p").lstrip("0")
            if scheduled_dt.date() == now_local.date():
                time_label = f"Today · {escape(time_display)}"
            else:
                time_label = scheduled_dt.strftime("%b %d, %Y · %I:%M %p").replace(" 0", " ")
                time_label = escape(time_label)

            title_lines: list[str] = []
            ticket_id = reminder.get("ticket_id")
            if ticket_id:
                title_lines.append(
                    (
                        '<span class="notification-card__title-line notification-card__title-line--primary">'
                        f'Ticket #{escape(ticket_id)}'
                        '</span>'
                    )
                )
            ont_id = reminder.get("ont_id")
            if ont_id:
                title_lines.append(
                    (
                        '<span class="notification-card__title-line notification-card__title-line--secondary">'
                        f'ONT ID: {escape(ont_id)}'
                        '</span>'
                    )
                )
            if not title_lines:
                title_lines.append(
                    '<span class="notification-card__title-line notification-card__title-line--primary">Reminder</span>'
                )
            header_text = "".join(title_lines)

            chip_parts: list[str] = []
            ticket_group = reminder.get("ticket_group")
            if ticket_group:
                group_color = _resolve_badge_color(TICKET_GROUP_BADGE_COLORS, ticket_group)
                style_attr = f' style="background:{group_color};color:#ffffff;"' if group_color else ""
                chip_parts.append(
                    f'<span class="notification-card__chip"{style_attr}>{escape(ticket_group)}</span>'
                )
            complaint_status = reminder.get("complaint_status")
            if complaint_status:
                status_color = _resolve_badge_color(COMPLAINT_STATUS_BADGE_COLORS, complaint_status)
                style_attr = f' style="background:{status_color};color:#ffffff;"' if status_color else ""
                chip_parts.append(
                    f'<span class="notification-card__chip"{style_attr}>{escape(complaint_status)}</span>'
                )
            ticket_type = reminder.get("ticket_type")
            if ticket_type:
                chip_parts.append(
                    f'<span class="notification-card__chip notification-card__chip--type">{escape(ticket_type)}</span>'
                )

            note_text = reminder.get("note", "")
            note_html = escape(note_text).replace("\n", "<br />") if note_text else ""

            card_html = [
                '<div class="notification-card">',
                '<div class="notification-card__header">',
                f'<div class="notification-card__title" style="color:#0f172a;">{header_text}</div>',
                f'<div class="notification-card__time">{time_label}</div>',
                '</div>',
            ]
            if chip_parts:
                card_html.append('<div class="notification-card__tags">' + "".join(chip_parts) + '</div>')
            if note_html:
                card_html.append(f'<div class="notification-card__note">{note_html}</div>')
            action_parts: list[str] = []
            if ticket_id:
                ticket_str = str(ticket_id)
                ticket_attr = escape(ticket_str)
                detail_query = {
                    "tab": "Call Tickets",
                    "subtab": "Tickets",
                    "ticket_detail": ticket_str,
                    "detail_token": f"{ticket_str}_{int(now_local.timestamp() * 1000)}",
                }
                detail_href = f"?{urlencode(detail_query)}"
                action_parts.append(
                    '<button type="button" class="notification-card__btn notification-card__btn--dismiss" '
                    f'data-ticket-id="{ticket_attr}" aria-disabled="true" data-disabled="true">Dismiss</button>'
                )
                action_parts.append(
                    f'<a href="{detail_href}" target="_self" class="notification-card__btn notification-card__btn--view" '
                    f'data-ticket-id="{ticket_attr}" onclick="try{{document.getElementById(\'notifications-toggle\').checked=false;}}catch(e){{}}">View Ticket</a>'
                )
            if action_parts:
                card_html.append('<div class="notification-card__actions">' + "".join(action_parts) + '</div>')
            card_html.append('</div>')
            notifications_content.append("".join(card_html))
        notifications_content.append('</div>')
    else:
        notifications_content.append('<div class="notifications-empty">No pending reminders.</div>')

    html.append('<div class="notifications-overlay">')
    html.append('<label for="notifications-toggle" class="overlay-backdrop"></label>')
    html.append('<div class="notifications-drawer">')
    html.append('<div class="menu-header"><div class="title">Notifications</div><label for="notifications-toggle" class="close-btn" aria-label="Close notifications"><span class="material-icons">close</span></label></div>')
    html.extend(notifications_content)
    html.append('</div></div>')

    html.append('</div>')

    st.markdown("".join(html), unsafe_allow_html=True)

# No-JS behavior handled via CSS; optional JS can be reintroduced if needed

# -------- Second-level tabs (Tickets / Settings) --------
SUBTAB_ICONS = {"Tickets": "toc", "Settings": "settings"}
SUBTAB_NAMES = list(SUBTAB_ICONS.keys())

# initialize active_subtab (default: Tickets)
if "active_subtab" not in st.session_state:
    st.session_state.active_subtab = "Tickets"

# allow switch via query param
if "subtab" in st.query_params:
    q = st.query_params["subtab"]
    if q in SUBTAB_NAMES:
        st.session_state.active_subtab = q

# Deep-links: open New Ticket on a specific tab and optionally prefill + autofill
try:
    if ("new" in st.query_params) and not st.session_state.get("_deep_link_done"):
        _nk = str(st.query_params.get("new", "")).strip().lower()
        _ont_q = st.query_params.get("ont")
        _af = str(st.query_params.get("autofill", "")).strip().lower()

        # Set base nav to open New Ticket area
        st.session_state.active_tab = "Call Tickets"
        st.session_state.active_subtab = "Tickets"
        st.session_state.show_new_form = True

        if _nk in ("complaint", "complaints"):
            st.session_state["_new_ticket_focus"] = "Complaints"
            if _ont_q is not None and str(_ont_q).strip():
                st.session_state["ont_c"] = str(_ont_q).strip()
                if _af in ("1", "true", "yes", "y"):
                    st.session_state["_do_autofill_c"] = True
        elif _nk in ("ai", "a&i", "activity", "activities", "inquiry", "inquiries", "activities & inquiries", "activities_and_inquiries", "activity / inquiry"):
            st.session_state["_new_ticket_focus"] = "Activities & Inquiries"
            if _ont_q is not None and str(_ont_q).strip():
                st.session_state["ont_ai"] = str(_ont_q).strip()
        elif _nk in ("osp", "appointment", "appointments", "osp appointment", "osp appointments", "osp_appointments"):
            st.session_state["_new_ticket_focus"] = "OSP Appointments"
            if _ont_q is not None and str(_ont_q).strip():
                st.session_state["ont_o"] = str(_ont_q).strip()
                if _af in ("1", "true", "yes", "y"):
                    st.session_state["_do_autofill_o"] = True
        st.session_state["_deep_link_done"] = True
        st.rerun()
except Exception:
    pass

# build subtabs HTML only for Call Tickets
if st.session_state.active_tab == "Call Tickets" and not _is_popup_view():
    base_tab_q = f"?tab={st.session_state.active_tab.replace(' ', '%20')}"
    sub_html = ['<div class="subtabs">']
    for name in SUBTAB_NAMES:
        icon = f'<span class="material-icons">{SUBTAB_ICONS[name]}</span>'
        cls = " sub-active" if st.session_state.active_subtab == name else ""
        sub_html.append(
            f'<a href="{base_tab_q}&subtab={name.replace(" ", "%20")}" target="_self" class="subtab{cls}">{icon} {name}</a>'
        )
    sub_html.append('</div>')
    st.markdown("".join(sub_html), unsafe_allow_html=True)

# ====================== UTILITIES ======================

def load_dim_options(filename: str, default=None, col_index: int = 1) -> list[str]:
    """
    Load options from the given Excel file's *second column* (col_index=1).
    Falls back to first column if the file has only one column,
    and to `default` if the file can't be read.
    """
    search_paths = [
        Path(filename),                 # same folder as app.py
        Path("./data") / filename,      # ./data/
        Path("/mnt/data") / filename,   # uploaded files path
    ]
    found_any = False
    for p in search_paths:
        if p.exists():
            found_any = True
            try:
                df = pd.read_excel(p)
                use_col = col_index if df.shape[1] > col_index else 0
                vals = (
                    df.iloc[:, use_col]
                    .dropna()
                    .astype(str)
                    .map(str.strip)
                )
                seen, out = set(), []
                for v in vals:
                    if v and v not in seen:
                        seen.add(v)
                        out.append(v)
                if out:
                    return out
            except Exception as e:
                logging.warning("Failed to load '%s' from '%s': %s. Using fallback list.", filename, p, e)
    if not found_any:
        logging.warning("Could not find '%s' in any search path. Using fallback list.", filename)
    return (default or [])


def load_implementation_note(fname: str) -> str:
    """Load a markdown implementation note from ./implementation_notes or repo root if present."""
    candidates = [Path("implementation_notes") / fname, Path(fname)]
    for p in candidates:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception:
            continue
    return ""

def add_ticket(row: dict):
    payload = _prepare_ticket_row(row)
    try:
        engine = get_tickets_engine()
        table = _reflect_tickets_table()
        insert_payload = _sanitize_ticket_payload_for_insert(payload)
        with engine.begin() as conn:
            conn.execute(insert(table).values(**insert_payload))
        _refresh_tickets_df(engine)
    except (SQLAlchemyError, Exception) as exc:  # pragma: no cover - Streamlit runtime logging
        logging.exception("Database insert failed: %s", exc)
        st.error("Database insert failed. Please try again after verifying the connection.")
        raise


# Load dropdown options from Excel (second column), with graceful fallbacks
ACTIVITY_TYPES         = load_dim_options("cx_dim_activity_inquiry_type.xlsx", ["General Inquiry", "Billing", "Technical", "Follow-up"])
CALL_TYPES             = load_dim_options("cx_dim_call_type.xlsx", ["Inbound", "Outbound", "Callback"])
COMPLAINT_TYPES        = load_dim_options("cx_dim_complaint_type.xlsx", ["Billing", "Connectivity", "Speed", "Other"])
EMP_SUGGESTION         = load_dim_options("cx_dim_employee_suggestion.xlsx", ["Escalate", "Schedule OSP", "Remote Fix", "Replace ONT"])
DEVICE_LOC             = load_dim_options("cx_dim_device_location.xlsx", ["Living Room", "Bedroom", "Office", "Other"])
ROOT_CAUSE             = load_dim_options("cx_dim_root_cause.xlsx", ["Power", "Fiber Cut", "Config", "Unknown"])
ONT_MODELS             = load_dim_options("cx_dim_ont_model.xlsx", ["ZTE F680", "Huawei HG8245", "Nokia XS-010X-Q"])
COMP_STATUS            = load_dim_options("cx_dim_complaint_status.xlsx", ["Open", "In Progress", "Pending Customer", "Closed"])
CITY_OPTIONS           = load_dim_options("cx_dim_city.xlsx", [])
ISSUE_TYPES            = load_dim_options("cx_dim_issue_type.xlsx", [])
KURDTEL_SERVICE_STATUS = load_dim_options("cx_dim_kurdtel_service_status.xlsx", [])
CREATED_BY_OPTIONS     = load_dim_options("cx_dim_assigned_to.xlsx", [])
FTTX_JOB_STATUS        = load_dim_options("cx_dim_fttx_job_status.xlsx", ["In Progress", "Completed", "Cancelled"])
CALLBACK_STATUS        = load_dim_options("cx_dim_callback_status.xlsx", [])
CALLBACK_REASON        = load_dim_options("cx_dim_callback_reason.xlsx", [])
FOLLOWUP_STATUS        = load_dim_options("cx_dim_followup_status.xlsx", [])
DIGICARE_ISSUES        = load_dim_options("cx_dim_digicare_issue.xlsx", [])
ONLINE_GAMES          = load_dim_options("cx_dim_online_game.xlsx", [])
REFUND_TYPES           = load_dim_options("cx_dim_refund_type.xlsx", ["Refund Info", "Refund Request"])
CHANNEL_OPTIONS        = load_dim_options("cx_dim_channel.xlsx", ["Call Center"])

DEFAULT_CREATED_BY = (
    CREATED_BY_OPTIONS[0]
    if CREATED_BY_OPTIONS and CREATED_BY_OPTIONS[0]
    else CURRENT_USER_NAME
)
if CURRENT_USER_NAME in CREATED_BY_OPTIONS:
    DEFAULT_CREATED_BY = CURRENT_USER_NAME


def _build_reminder_recipient_options() -> list[str]:
    seen = set()
    options = []
    if REMINDER_ALL_VALUE:
        options.append(REMINDER_ALL_VALUE)
        seen.add(REMINDER_ALL_VALUE)
    for opt in CREATED_BY_OPTIONS:
        label = str(opt).strip()
        if label and label not in seen:
            options.append(label)
            seen.add(label)
    if DEFAULT_CREATED_BY and DEFAULT_CREATED_BY not in seen:
        options.append(DEFAULT_CREATED_BY)
        seen.add(DEFAULT_CREATED_BY)
    return options


REMINDER_RECIPIENT_OPTIONS = _build_reminder_recipient_options()


def _normalize_reminder_selection(values) -> list[str]:
    if not values:
        return []
    normalized = []
    for raw in values:
        if raw is None:
            continue
        label = str(raw).strip()
        if not label or label.lower() == "none":
            continue
        if label == REMINDER_ALL_VALUE:
            return [REMINDER_ALL_VALUE]
        if label not in normalized:
            normalized.append(label)
    return normalized


def _parse_reminder_recipients(value) -> list[str]:
    if isinstance(value, list):
        base = value
    else:
        text = str(value or "")
        if not text:
            base = []
        else:
            base = [chunk.strip() for chunk in text.replace(",", ";").split(";")]
    return _normalize_reminder_selection(base)


def _serialize_reminder_recipients(values) -> str:
    normalized = _normalize_reminder_selection(values)
    if not normalized:
        return ""
    if len(normalized) == 1 and normalized[0] == REMINDER_ALL_VALUE:
        return REMINDER_ALL_VALUE
    return "; ".join(normalized)


def _ensure_recipient_options(options: list[str], selection: list[str]) -> list[str]:
    merged = list(options)
    for val in selection:
        label = str(val).strip()
        if label and label not in merged:
            merged.append(label)
    return merged



# OSP types (removed "Sub-Districts Interface")
OSP_TYPES = [
    "No Power", "Fiber Cut", "Fast Connector", "Relocate ONT",
    "Degraded", "Rearrange Fiber", "Closure", "Manhole", "Fiber", "Pole"
]

# FTTG fixed options
FTTG_OPTIONS = ["Yes", "No"]

# ====================== EDIT HELPERS ======================
def _get_row_by_id(_id: int):
    df = st.session_state.tickets_df
    if df.empty or "id" not in df.columns:
        return None, None
    idx = df.index[df["id"] == _id]
    if len(idx) == 0:
        return None, None
    i = idx[0]
    return i, df.loc[i].to_dict()

def _update_row(i, patch: dict):
    df = st.session_state.tickets_df
    if df.empty or "id" not in df.columns:
        return

    ticket_id = None
    if i is not None and 0 <= i < len(df):
        raw_id = df.iloc[i]["id"]
        if pd.notna(raw_id):
            try:
                ticket_id = int(raw_id)
            except Exception:
                ticket_id = raw_id

    if ticket_id is None:
        return

    sanitized = _sanitize_ticket_update(patch)
    if not sanitized:
        return

    try:
        engine = get_tickets_engine()
        table = _reflect_tickets_table()
        with engine.begin() as conn:
            conn.execute(update(table).where(table.c.id == ticket_id).values(**sanitized))
        _refresh_tickets_df(engine)
    except (SQLAlchemyError, Exception) as exc:  # pragma: no cover - Streamlit runtime logging
        logging.exception("Database update failed: %s", exc)
        st.error("Database update failed. Please verify the connection and retry.")
        raise

# ---- New Ticket helpers ----
def _reset_new_ticket_fields():
    """Clear new-ticket form fields and show the new ticket form."""
    st.session_state.show_new_form = True
    for _k in [
        # Activities & Inquiries
        'ont_ai','ont_ai_locked','ai_pending_clear','autofill_message_ai','autofill_level_ai','description_ai','channel_ai','outage_start_ai','outage_end_ai',
    'sb_type_of_activity_inquiries_1','sb_digicare_issue_1','sb_call_type_1',
        # Complaints
        'ont_c','olt_c','olt_c_filled','ont_c_locked','c_pending_clear','autofill_message_c','autofill_level_c','description_c',
    'sb_type_of_complaint_1','sb_refund_type_1','sb_complaint_status_1','sb_call_type_2','sb_employee_suggestion_1','sb_root_cause_1','ont_model','sb_device_location_1','channel_c',
    'kurdtel_status_c','online_game_c','online_game_other_c','outage_start_c','outage_end_c','sip_c_value','name_c_value','kurdtel_device_type_c_value','issue_type_kurdtel','second_number_c',
    'reminder_recipient_c','reminder_note_c','reminder_date_c','reminder_time_c',
        # OSP
    'ont_o','ont_o_locked','osp_pending_clear','autofill_message_o','autofill_level_o','city_o','fttg_o','address_o','olt_o','line_card_o','gpon_o','issue_type_o','description_o','channel_o',
    'sb_call_type_3','sb_osp_appointment_type_1','second_number_o',
    ]:
        try:
            st.session_state[_k] = ''
        except Exception:
            pass
    # Ensure date/time widgets start unset to avoid type mismatches
    for _k in ('reminder_date_c','reminder_time_c','outage_start_ai','outage_end_ai'):
        try:
            st.session_state.pop(_k, None)
        except Exception:
            pass
    st.session_state['reminder_enabled_c'] = False
    st.session_state['reminder_recipient_c'] = [DEFAULT_CREATED_BY] if DEFAULT_CREATED_BY else []
    st.session_state['reminder_note_c'] = ''
    # Ensure Call Type defaults to the first option (e.g., 'Inbound') on fresh form
    try:
        if CALL_TYPES:
            st.session_state['sb_call_type_1'] = CALL_TYPES[0]
            st.session_state['sb_call_type_2'] = CALL_TYPES[0]
            st.session_state['sb_call_type_3'] = CALL_TYPES[0]
        else:
            for _k in ['sb_call_type_1','sb_call_type_2','sb_call_type_3']:
                st.session_state.pop(_k, None)
    except Exception:
        pass

    try:
        if CHANNEL_OPTIONS:
            st.session_state['channel_ai'] = CHANNEL_OPTIONS[0]
            st.session_state['channel_c'] = CHANNEL_OPTIONS[0]
            st.session_state['channel_o'] = CHANNEL_OPTIONS[0]
        else:
            for _k in ['channel_ai', 'channel_c', 'channel_o']:
                st.session_state.pop(_k, None)
    except Exception:
        pass

def _is_new_form_dirty() -> bool:
    """Return True if any known new-ticket fields have values indicating in-progress input."""
    candidates = [
        # AI
    'ont_ai', 'sb_type_of_activity_inquiries_1', 'sb_digicare_issue_1', 'sb_call_type_1', 'channel_ai', 'outage_start_ai', 'outage_end_ai',
        # Complaints
        'ont_c', 'sb_type_of_complaint_1', 'sb_refund_type_1', 'sb_complaint_status_1', 'sb_call_type_2',
        'sb_employee_suggestion_1', 'sb_root_cause_1', 'ont_model', 'sb_device_location_1',
        'olt_c', 'kurdtel_status_c', 'online_game_c', 'online_game_other_c', 'outage_start_c', 'outage_end_c',
    'ip_c', 'vlan_c', 'packet_loss_c', 'high_ping_c', 'channel_c',
        'reminder_enabled_c', 'reminder_note_c', 'reminder_date_c', 'reminder_time_c',
        # OSP
    'ont_o', 'sb_osp_appointment_type_1', 'sb_call_type_3', 'second_number_o', 'fttg_o', 'city_o', 'address_o', 'channel_o',
    ]
    for k in candidates:
        v = st.session_state.get(k)
        # Treat default Call Type (first option) as not-dirty
        if k in ('sb_call_type_1', 'sb_call_type_2', 'sb_call_type_3') and CALL_TYPES:
            if v in (None, '', [], {}, 0, CALL_TYPES[0]):
                continue
        if v not in (None, '', [], {}, 0):
            return True
    return False

@st.dialog("Start a new ticket?")
def _confirm_new_ticket_dialog():
    st.markdown('<div class="confirm-new-ticket">', unsafe_allow_html=True)
    st.write("You have unsaved changes. Starting a new ticket will discard them. Continue?")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Return", key="cancel_start_new_ticket", use_container_width=True):
            st.session_state['_confirm_new_ticket'] = False
            st.rerun()
    with c2:
        if st.button("Start new ticket", key="confirm_start_new_ticket", use_container_width=True):
            _reset_new_ticket_fields()
            st.session_state['_confirm_new_ticket'] = False
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


@st.dialog("New ticket created")
def _ticket_saved_dialog():
    message = st.session_state.get('_ticket_saved_message') or "New ticket created successfully."
    st.markdown('<div class="ticket-saved-dialog">', unsafe_allow_html=True)
    st.write(message)
    if st.button("Close window", key="close_ticket_popup", use_container_width=True):
        st.session_state['_ticket_saved_message'] = message
        st.session_state['_show_ticket_saved_dialog'] = False
        _schedule_popup_close()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


@st.dialog("View / Edit Ticket")
def edit_ticket_dialog(ticket_id: int):
    i, row = _get_row_by_id(ticket_id)
    if row is None:
        st.error("Ticket not found.")
        return

    st.caption(f"Ticket ID: {row.get('id')} · Created: {row.get('created_at')} · Group: {row.get('ticket_group')}")

    with st.form(f"edit_form_{ticket_id}"):
        g = row.get("ticket_group")

        # Common fields across groups
        ont = st.text_input("ONT ID", value=str(row.get("ont_id") or ""))
        current_call_type = row.get("call_type")
        call_type_index = CALL_TYPES.index(current_call_type) if (CALL_TYPES and current_call_type in CALL_TYPES) else None
        type_call = st.selectbox("Call Type", CALL_TYPES, index=call_type_index, placeholder="")
        _creator_options = CREATED_BY_OPTIONS or []
        _creator_index = None
        if row.get("created_by") in _creator_options:
            _creator_index = _creator_options.index(row.get("created_by"))
        created_by_value = st.selectbox("Created By", _creator_options, index=_creator_index, placeholder="")
        channel_index = None
        if CHANNEL_OPTIONS and row.get("channel") in CHANNEL_OPTIONS:
            channel_index = CHANNEL_OPTIONS.index(row.get("channel"))
        channel_value = st.selectbox(
            "Channel",
            CHANNEL_OPTIONS,
            index=channel_index if CHANNEL_OPTIONS else None,
            placeholder="",
        ) if CHANNEL_OPTIONS else (row.get("channel") or "")

        # Group-specific
        updates = {}
        updates["channel"] = channel_value
        if g == "Activities & Inquiries":
            act = st.selectbox("Activity / Inquiry Type", ACTIVITY_TYPES, index=(ACTIVITY_TYPES.index(row["activity_inquiry_type"]) if row.get("activity_inquiry_type") in ACTIVITY_TYPES else None), placeholder="")
            desc = st.text_area("Description", value=str(row.get("description") or ""), height=120)
            updates.update({
                "activity_inquiry_type": act,
                "description": desc,
            })
        elif g == "Complaints":
            comp = st.selectbox("Complaint Type", COMPLAINT_TYPES, index=(COMPLAINT_TYPES.index(row["complaint_type"]) if row.get("complaint_type") in COMPLAINT_TYPES else None), placeholder="")
            refund_t = ""
            if comp == "Refund":
                refund_t = st.selectbox(
                    "Refund Type", REFUND_TYPES,
                    index=(REFUND_TYPES.index(row.get("refund_type")) if row.get("refund_type") in REFUND_TYPES else None),
                    placeholder=""
                ) or ""
            emp = st.selectbox("Employee Suggestion", EMP_SUGGESTION, index=(EMP_SUGGESTION.index(row["employee_suggestion"]) if row.get("employee_suggestion") in EMP_SUGGESTION else None), placeholder="")
            status = st.selectbox("Complaint Status", COMP_STATUS, index=(COMP_STATUS.index(row["complaint_status"]) if row.get("complaint_status") in COMP_STATUS else None), placeholder="")
            root = st.selectbox("Root Cause", ROOT_CAUSE, index=(ROOT_CAUSE.index(row["root_cause"]) if row.get("root_cause") in ROOT_CAUSE else None), placeholder="")
            model = st.selectbox("ONT Model", ONT_MODELS, index=(ONT_MODELS.index(row["ont_model"]) if row.get("ont_model") in ONT_MODELS else None), placeholder="")
            devloc = st.selectbox("Device Location", DEVICE_LOC, index=(DEVICE_LOC.index(row["device_location"]) if row.get("device_location") in DEVICE_LOC else None), placeholder="")
            olt = st.text_input("OLT", value=str(row.get("olt") or ""))
            second = st.text_input("Second Number", value=str(row.get("second_number") or ""))
            desc = st.text_area("Description", value=str(row.get("description") or ""), height=120)
            # Outage dates (for Refund complaints with Refund Type = Refund Request)
            outage_start_e = None
            outage_end_e = None
            try:
                if (comp == "Refund") and (refund_t == "Refund Request"):
                    try:
                        _s = row.get("outage_start_date")
                        outage_start_e = pd.to_datetime(_s).date() if _s else _now_local().date()
                    except Exception:
                        outage_start_e = _now_local().date()
                    try:
                        _e = row.get("outage_end_date")
                        outage_end_e = pd.to_datetime(_e).date() if _e else _now_local().date()
                    except Exception:
                        outage_end_e = _now_local().date()
                    outage_start_e = st.date_input("Outage Start Date", value=outage_start_e)
                    outage_end_e = st.date_input("Outage End Date", value=outage_end_e)
            except Exception:
                outage_start_e = None
                outage_end_e = None
            kurdtel = ""
            if row.get("complaint_type") == "Kurdtel":
                kurdtel = st.selectbox("Kurdtel Service Status", KURDTEL_SERVICE_STATUS, index=(KURDTEL_SERVICE_STATUS.index(row["kurdtel_service_status"]) if row.get("kurdtel_service_status") in KURDTEL_SERVICE_STATUS else None), placeholder="")
            online_game_e = ""
            if row.get("complaint_type") == "Online Game Issue":
                online_game_e = st.selectbox("Online Game", ONLINE_GAMES, index=(ONLINE_GAMES.index(row.get("online_game")) if row.get("online_game") in ONLINE_GAMES else None), placeholder="")
            # Callback / Follow-up fields
            cb_status = st.selectbox(
                "Call-Back Status", CALLBACK_STATUS,
                index=(CALLBACK_STATUS.index(row["callback_status"]) if row.get("callback_status") in CALLBACK_STATUS else None),
                placeholder=""
            )
            cb_reason = st.selectbox(
                "Call-Back Reason", CALLBACK_REASON,
                index=(CALLBACK_REASON.index(row["callback_reason"]) if row.get("callback_reason") in CALLBACK_REASON else None),
                placeholder=""
            )
            fu_status = st.selectbox(
                "Follow-Up Status", FOLLOWUP_STATUS,
                index=(FOLLOWUP_STATUS.index(row["followup_status"]) if row.get("followup_status") in FOLLOWUP_STATUS else None),
                placeholder=""
            )
            reminder_enabled_edit_default = bool(row.get("reminder_enabled"))
            existing_reminder_recipient = row.get("reminder_recipient") or ""
            reminder_note_edit = row.get("reminder_note") or ""
            reminder_at_existing = row.get("reminder_at") or ""
            reminder_dt_existing = None
            try:
                if reminder_at_existing:
                    parsed_dt = pd.to_datetime(reminder_at_existing, errors="coerce")
                    if pd.notna(parsed_dt):
                        reminder_dt_existing = _to_local_naive(parsed_dt.to_pydatetime())
            except Exception:
                reminder_dt_existing = None
            rem_date_edit = None
            rem_time_edit = None
            recipient_state_key = f"reminder_recipient_edit_{ticket_id}"
            if recipient_state_key not in st.session_state:
                _initial_selection = _parse_reminder_recipients(existing_reminder_recipient)
                if not _initial_selection and DEFAULT_CREATED_BY:
                    _initial_selection = [DEFAULT_CREATED_BY]
                st.session_state[recipient_state_key] = _initial_selection
            st.session_state[recipient_state_key] = _normalize_reminder_selection(
                st.session_state.get(recipient_state_key, [])
            )
            selected_recipients_edit = st.session_state.get(recipient_state_key, [])
            reminder_enabled_edit = st.checkbox("Set Reminder", value=reminder_enabled_edit_default)
            selected_recipients_edit = _normalize_reminder_selection(
                st.session_state.get(recipient_state_key, [])
            )
            if reminder_enabled_edit:
                st.markdown(
                    """
                    <div class=\"reminder-panel\" style=\"margin-top:0.5rem;padding:1rem;border-radius:0.75rem;border:1px solid rgba(100,116,139,0.35);background:rgba(226,232,240,0.35);\">
                    """,
                    unsafe_allow_html=True,
                )
                rem_cols = st.columns(3)
                with rem_cols[0]:
                    recipient_options_edit = _ensure_recipient_options(
                        REMINDER_RECIPIENT_OPTIONS,
                        st.session_state.get(recipient_state_key, []),
                    )
                    _selected_edit_raw = st.multiselect(
                        "Reminder Recipient",
                        recipient_options_edit,
                        key=recipient_state_key,
                        placeholder="Select recipient(s)…",
                    )
                    selected_recipients_edit = _normalize_reminder_selection(
                        _selected_edit_raw
                    )
                with rem_cols[1]:
                    rem_date_default = reminder_dt_existing.date() if reminder_dt_existing is not None else _now_local().date()
                    rem_date_edit = st.date_input("Reminder Date", value=rem_date_default)
                with rem_cols[2]:
                    rem_time_default = (
                        reminder_dt_existing.time().replace(microsecond=0)
                        if reminder_dt_existing is not None else _now_local().time().replace(second=0, microsecond=0)
                    )
                    rem_time_edit = st.time_input("Reminder Time", value=rem_time_default)
                reminder_note_edit = st.text_area("Reminder Note", value=reminder_note_edit, height=80)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                selected_recipients_edit = []
                reminder_note_edit = ""
            reminder_at_edit_str = ""
            if reminder_enabled_edit and rem_date_edit and rem_time_edit:
                try:
                    reminder_at_edit_str = datetime.combine(rem_date_edit, rem_time_edit).isoformat()
                except Exception:
                    reminder_at_edit_str = ""
            reminder_recipient_edit_str = _serialize_reminder_recipients(selected_recipients_edit)
            updates.update({
                "complaint_type": comp,
                "refund_type": refund_t if comp == "Refund" else "",
                "employee_suggestion": emp,
                "complaint_status": status,
                "root_cause": root,
                "ont_model": model,
                "device_location": devloc,
                "olt": olt,
                "second_number": second,
                "description": desc,
                "outage_start_date": (outage_start_e.isoformat() if hasattr(outage_start_e, 'isoformat') else (str(outage_start_e) if outage_start_e else "")),
                "outage_end_date": (outage_end_e.isoformat() if hasattr(outage_end_e, 'isoformat') else (str(outage_end_e) if outage_end_e else "")),
                "kurdtel_service_status": kurdtel if row.get("complaint_type") == "Kurdtel" else "",
                "online_game": online_game_e if row.get("complaint_type") == "Online Game Issue" else "",
                "callback_status": cb_status,
                "callback_reason": cb_reason,
                "followup_status": fu_status,
                "reminder_enabled": reminder_enabled_edit,
                "reminder_recipient": reminder_recipient_edit_str if reminder_enabled_edit else "",
                "reminder_note": reminder_note_edit if reminder_enabled_edit else "",
                "reminder_at": reminder_at_edit_str,
            })
        elif g == "OSP Appointments":
            osp = st.selectbox("OSP Appointment Type", OSP_TYPES, index=(OSP_TYPES.index(row["osp_type"]) if row.get("osp_type") in OSP_TYPES else None), placeholder="")
            city = st.selectbox("City", CITY_OPTIONS, index=(CITY_OPTIONS.index(row["city"]) if row.get("city") in CITY_OPTIONS else None), placeholder="")
            issue = st.selectbox("Issue Type", ISSUE_TYPES, index=(ISSUE_TYPES.index(row["issue_type"]) if row.get("issue_type") in ISSUE_TYPES else None), placeholder="")
            fttg = st.selectbox("FTTG", FTTG_OPTIONS, index=(FTTG_OPTIONS.index(row["fttg"]) if row.get("fttg") in FTTG_OPTIONS else None), placeholder="")
            second = st.text_input("Second Number", value=str(row.get("second_number") or ""))
            addr = st.text_area("Address", value=str(row.get("address") or ""), height=90)
            desc = st.text_area("Description", value=str(row.get("description") or ""), height=120)
            # FTTX fields (editable in edit dialog)
            fttx_status_e = st.selectbox(
                "FTTX Job Status", FTTX_JOB_STATUS,
                index=(FTTX_JOB_STATUS.index(row["fttx_job_status"]) if row.get("fttx_job_status") in FTTX_JOB_STATUS else 0),
                placeholder=""
            )
            fttx_remarks_e = st.text_area("FTTX Job Remarks", value=str(row.get("fttx_job_remarks") or ""), height=80)
            st.text_input("FTTX Cancel Reason", value=str(row.get("fttx_cancel_reason") or ""), disabled=True)
            updates.update({
                "osp_type": osp,
                "city": city,
                "issue_type": issue,
                "fttg": fttg,
                "second_number": second,
                "address": addr,
                "fttx_job_status": fttx_status_e,
                "fttx_job_remarks": fttx_remarks_e,
                "fttx_cancel_reason": row.get("fttx_cancel_reason") or "",
                "description": desc,
            })
        else:
            # Fallback: show description
            desc = st.text_area("Description", value=str(row.get("description") or ""), height=120)
            updates.update({"description": desc})

        save = st.form_submit_button("Save changes")
        if save:
            base_updates = {
                "ont_id": ont,
                "call_type": type_call,
                "created_by": created_by_value or DEFAULT_CREATED_BY,
            }
            base_updates.update(updates)
            _update_row(i, base_updates)
            st.session_state['_edit_open_id'] = None
            st.success("Ticket updated.")
            st.rerun()

# ====================== PAGE CONTENT ======================
# If user chose Settings subtab (only within Call Center Tickets), show settings UI
if st.session_state.active_tab == "Call Tickets" and st.session_state.active_subtab == "Settings":
    st.markdown("### Manage Lookups")
    # description removed per request

    DROPDOWN_CONFIGS = [
        ("Employee Suggestion", "EMP_SUGGESTION"),
        ("Device Location", "DEVICE_LOC"),
        ("Root Cause", "ROOT_CAUSE"),
        ("ONT Model", "ONT_MODELS"),
        ("Complaint Status", "COMP_STATUS"),
        ("Refund Type", "REFUND_TYPES"),
        ("Online Game", "ONLINE_GAMES"),
    ("Channel", "CHANNEL_OPTIONS"),
        ("City", "CITY_OPTIONS"),
        ("Issue Type", "ISSUE_TYPES"),
    ("Kurdtel Service Status", "KURDTEL_SERVICE_STATUS"),
    ("Created By", "CREATED_BY_OPTIONS"),
        ("FTTX Job Status", "FTTX_JOB_STATUS"),
        ("Callback Status", "CALLBACK_STATUS"),
        ("Callback Reason", "CALLBACK_REASON"),
        ("Followup Status", "FOLLOWUP_STATUS"),
    ]

    for label, key in DROPDOWN_CONFIGS:
        init_list_name = f"settings_{key}"
        if init_list_name not in st.session_state:
            st.session_state[init_list_name] = list(globals().get(key, []) or [])

        vals = st.session_state[init_list_name]

        # per-config message storage key (used to show messages under the value column)
        base_msg_key = f"{key}_message"

        with st.expander(label, expanded=False):
            if not vals:
                st.info("No items defined yet.")

            # Add-new row: input first (left), action button on the right
            new_key = f"{key}_new"
            # ensure the input buffer exists without overwriting an existing widget value
            st.session_state.setdefault(new_key, "")
            # if a previous action requested the input be cleared, do it before the widget is created
            clear_flag = f"{new_key}_clear"
            if st.session_state.get(clear_flag):
                st.session_state[new_key] = ""
                try:
                    del st.session_state[clear_flag]
                except Exception:
                    pass
            # input column left, action column right
            add_input_col, add_action_col = st.columns([11, 1])
            with add_input_col:
                # non-empty hidden label to avoid Streamlit label warnings
                st.text_input(f"add_{key}", key=new_key, placeholder=f"Add new value to {label}", label_visibility="collapsed")

                # show add-action message under the input column (if any)
                _msg = st.session_state.get(base_msg_key)
                if _msg:
                    _lvl, _txt = _msg
                    if _lvl == "error":
                        st.error(_txt)
                    elif _lvl == "warning":
                        st.warning(_txt)
                    else:
                        st.success(_txt)

            with add_action_col:
                # intrinsic-size button on the right
                if st.button("➕", key=f"{key}_add_btn", help=f"Add new to {label}", use_container_width=False):
                    nv = (st.session_state.get(new_key) or "").strip()
                    if not nv:
                        st.session_state[base_msg_key] = ("error", "Enter a non-empty value.")
                    elif nv in st.session_state[init_list_name]:
                        st.session_state[base_msg_key] = ("warning", "Value already exists.")
                    else:
                        st.session_state[init_list_name].append(nv)
                        # request that the add-input be cleared on the next rerun (safe: do not overwrite widget value now)
                        st.session_state[f"{new_key}_clear"] = True
                        st.session_state[base_msg_key] = ("success", "Added.")
                        st.rerun()

            # Render each existing value: action (left, narrow), value (right, wide)
            for i in range(len(vals)):
                edit_flag = f"{key}_editing_{i}"
                edit_buf = f"{key}_edit_{i}"
                item_msg_key = f"{key}_message_{i}"
                if edit_flag not in st.session_state:
                    st.session_state[edit_flag] = False
                # set default buffer value without causing widget/session-state conflict
                st.session_state.setdefault(edit_buf, vals[i])

                # value column first (left), action on the right
                value_col, action_col = st.columns([11, 1], gap="small")
                with value_col:
                    if st.session_state[edit_flag]:
                        # avoid passing value= when session state already exists
                        st.text_input(f"edit_{key}_{i}", key=edit_buf, label_visibility="collapsed")
                    else:
                        # show value inside a rounded, gray box so it looks disabled/read-only
                        # expand the value box to fill the column so it matches the Add-row input width
                        st.markdown(
                            f"<div style='background:#f3f4f6;color:#374151;padding:8px 12px;border-radius:8px;margin:4px 0;display:block;width:100%;box-sizing:border-box;font-size:0.95rem;white-space:normal;overflow-wrap:break-word'>{vals[i]}</div>",
                            unsafe_allow_html=True,
                        )

                    # show per-item message under the value column (if any)
                    _imsg = st.session_state.get(item_msg_key)
                    if _imsg:
                        _ilvl, _itxt = _imsg
                        if _ilvl == "error":
                            st.error(_itxt)
                        elif _ilvl == "warning":
                            st.warning(_itxt)
                        else:
                            st.success(_itxt)

                with action_col:
                    # keep intrinsic button footprint so size is stable (no use_container_width)
                    if st.session_state[edit_flag]:
                        if st.button("💾", key=f"{key}_save_{i}", help="Save", use_container_width=False):
                            nv = (st.session_state.get(edit_buf) or "").strip()
                            if nv:
                                if i < len(st.session_state[init_list_name]):
                                    st.session_state[init_list_name][i] = nv
                                else:
                                    st.session_state[init_list_name].append(nv)
                                st.session_state[item_msg_key] = ("success", "Saved.")
                            else:
                                st.session_state[item_msg_key] = ("error", "Enter a non-empty value.")
                            st.session_state[edit_flag] = False
                            st.rerun()
                    else:
                        if st.button("✏️", key=f"{key}_editbtn_{i}", help="Edit", use_container_width=False):
                            st.session_state[edit_flag] = True
                            st.session_state[edit_buf] = vals[i]
                            # clear any existing item message when entering edit mode
                            st.session_state.pop(item_msg_key, None)
                            st.rerun()
    # prevent the rest of the page (Tickets UI / grid) from rendering when on Settings
    st.stop()
# original main content
if st.session_state.active_tab == "Client":
    # Client page layout: two columns
    st.markdown('<div id="client-layout">', unsafe_allow_html=True)
    left, right = st.columns([2, 5], gap=None)
    with left:
        # Native Streamlit version of the client card (no custom HTML)
        st.subheader("Dahi Nemutlu")

        client_rows = [
            ("ID", "22000"),
            ("Client Type", "Employee"),
            ("Account Number", "07700000000"),
            ("Service ID", "91000000"),
            ("Sip", "This Client has no Sip."),
            ("City", ""),
            ("Address", ""),
            ("Comment", ""),
            ("Created At", ""),
            ("Last Change", ""),
        ]

        # Use compact key/value rows so the column's intrinsic width stays small
        for idx, (lbl, val) in enumerate(client_rows):
            lcol, vcol = st.columns([1, 2], gap="small")
            with lcol:
                st.caption(lbl)
            with vcol:
                st.write(val or "—")
            # Compact separator between rows (skip after the last row)
            if idx < len(client_rows) - 1:
                st.markdown('<hr style="margin:0;border:0;border-top:1px solid rgba(0,0,0,0.10);" />', unsafe_allow_html=True)
    with right:
        # Replace Streamlit tabs with HTML subtabs similar to "Call Tickets"
        CLIENT_SUBTAB_ICONS = {
            "CPEs": "router",
            "Assign": "assignment_turned_in",
            "SIP": "dialer_sip",
            "Client Operations": "assignment",
            "Client Attachments": "attachment",
            "Client Attachments KYC": "fingerprint",
            "Dicigare Tickets": "report_problem",
            "Call Tickets": "phone",
        }
        CLIENT_SUBTAB_NAMES = list(CLIENT_SUBTAB_ICONS.keys())

        # initialize active client subtab
        if "active_client_subtab" not in st.session_state:
            st.session_state.active_client_subtab = "CPEs"
        # allow switch via query param only when on Client tab
        if st.session_state.active_tab == "Client" and "client_subtab" in st.query_params:
            cq = st.query_params["client_subtab"]
            if cq in CLIENT_SUBTAB_NAMES:
                st.session_state.active_client_subtab = cq

        base_q = "?tab=Client"
        sub_html = ['<div class="subtabs" style="margin-left:12px;">']
        for i, name in enumerate(CLIENT_SUBTAB_NAMES):
            icon = f'<span class="material-icons">{CLIENT_SUBTAB_ICONS[name]}</span>'
            cls = " sub-active" if st.session_state.active_client_subtab == name else ""
            if i == 0:
                # Only the first subtab is clickable
                sub_html.append(
                    f'<a href="{base_q}&client_subtab={name.replace(" ", "%20")}" target="_self" class="subtab{cls}">{icon} {name}</a>'
                )
            else:
                # Render others as disabled/non-clickable
                sub_html.append(
                    f'<span class="subtab subtab-disabled{cls}" title="Disabled">{icon} {name}</span>'
                )
        sub_html.append('</div>')
        st.markdown("".join(sub_html), unsafe_allow_html=True)

        # Right column content based on active client subtab
        current = st.session_state.active_client_subtab
        if current == "CPEs":
            _ensure_popup_script()
            ont_id = "1400"
            popup_base = "?popup=call_ticket&tab=Call%20Tickets&subtab=Tickets"
            activity_url = f"{popup_base}&new=activity&ont={ont_id}"
            complaint_url = f"{popup_base}&new=complaint&ont={ont_id}&autofill=1"
            osp_url = f"{popup_base}&new=osp%20appointment&ont={ont_id}&autofill=1"
            activity_href = activity_url.replace("&", "&amp;")
            complaint_href = complaint_url.replace("&", "&amp;")
            osp_href = osp_url.replace("&", "&amp;")
            st.markdown(
                f"""
                <div class="exp-card">
                    <details open>
                        <summary>#62000 - ccbe.5991.0000 - {ont_id} - Employee-1 - <span class="dt-green">2026-12-31 23:59:50</span></summary>
                        <div class="cpe-actions">
                            <div class="dropdown" style="position:relative;display:inline-block;">
                                <input type="checkbox" id="nt-dropdown-toggle" class="nt-dropdown-toggle" />
                                <label for="nt-dropdown-toggle" class="cpe-btn cpe-dropdown-btn" aria-haspopup="true" aria-expanded="false"><span class="material-icons">add_ic_call</span> New Call Ticket <span class="material-icons caret">expand_more</span></label>
                                <div class="dropdown-menu" style="position:absolute;top:100%;left:0;background:#fff;border:1px solid #E5E7EB;border-radius:.5rem;box-shadow:0 4px 16px rgba(0,0,0,0.1);min-width:220px;padding:.25rem 0;z-index:10;">
                                    <a class="dropdown-item" style="display:block;padding:.5rem .75rem;color:#023058;text-decoration:none;" href="{activity_href}" data-call-ticket-popup="true" data-popup-href="{activity_href}" data-toggle-id="nt-dropdown-toggle" data-confirm-existing-popup="1">Activity / Inquiry</a>
                                    <a class="dropdown-item" style="display:block;padding:.5rem .75rem;color:#023058;text-decoration:none;" href="{complaint_href}" data-call-ticket-popup="true" data-popup-href="{complaint_href}" data-toggle-id="nt-dropdown-toggle" data-confirm-existing-popup="1">Complaint</a>
                                    <a class="dropdown-item" style="display:block;padding:.5rem .75rem;color:#023058;text-decoration:none;" href="{osp_href}" data-call-ticket-popup="true" data-popup-href="{osp_href}" data-toggle-id="nt-dropdown-toggle" data-confirm-existing-popup="1">OSP Appointment</a>
                                </div>
                                <label for="nt-dropdown-toggle" class="cpe-dd-overlay" aria-hidden="true"></label>
                            </div>
                            <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">settings_remote</span> Remote Access</span>
                            <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">restart_alt</span> Restart Session</span>
                            <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">remove_circle</span> Unblock</span>
                            <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">description</span> Request</span>
                            <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">public</span> Public Ip</span>
                            <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">credit_card</span> Recharge</span>
                            <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">undo</span> Undo Recharge</span>
                            <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">near_me</span> Transfer</span>
                            <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">find_replace</span> Replace</span>
                            <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">highlight_off</span> Un-Assign</span>
                        </div>
                        <div class="exp-content">
                            <div class="kv-cols">
                                <div class="kv-list">
                                    <div class="kv-row"><div class="kv-label">ID</div><div class="kv-value">62756</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">Phone</div><div class="kv-value">07700000000</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">ONT Model</div><div class="kv-value">844G</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">Package</div><div class="kv-value">Employee-1</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">Expiration</div><div class="kv-value">2026-12-31 23:59:50</div></div>
                                </div>
                                <div class="kv-list">
                                    <div class="kv-row"><div class="kv-label">OLT</div><div class="kv-value">NTWK-Sul-Pasha-OLT-00</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">ONT ID</div><div class="kv-value">{ont_id}</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">Serial</div><div class="kv-value">CXNK00000000</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">MAC</div><div class="kv-value">ccbe.5991.0000</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">Line Card</div><div class="kv-value">Cisco NCS 5500</div></div>
                                </div>
                                <div class="kv-list">
                                    <div class="kv-row"><div class="kv-label">Operational Status</div><div class="kv-value">enable</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">Status</div><div class="kv-value">Online</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">IP</div><div class="kv-value">10.49.72.000</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">VLAN</div><div class="kv-value">3021</div></div>
                                    <div class="kv-sep"></div>
                                    <div class="kv-row"><div class="kv-label">GPON</div><div class="kv-value">2.5G/1.25G</div></div>
                                </div>
                            </div>
                        </div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Developer notes: render client markdown below the CPE card inside an expander
            # Add a bit of top spacing and indent using columns so it doesn't stick to the left/top
            try:
                _dev_md_client = load_implementation_note("client.md")
                if _dev_md_client:
                    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                    pad_l, content, pad_r = st.columns([1, 100, 10])
                    with content:
                        with st.expander("Implementation notes (Client)", expanded=False):
                            st.markdown('<div class="implementation-notes-tag"></div>', unsafe_allow_html=True)
                            st.markdown(_dev_md_client)
            except Exception:
                pass
        elif current == "Assign":
            st.write("")
        elif current == "SIP":
            st.write("")
        elif current == "Client Operations":
            st.write("")
        elif current == "Client Attachments":
            st.write("")
        elif current == "Client Attachments KYC":
            st.write("")
        elif current == "Dicigare Tickets":
            st.write("")
        elif current == "Call Tickets":
            st.write("")
    st.markdown('</div>', unsafe_allow_html=True)
elif st.session_state.active_tab != "Call Tickets":
    st.write("Content for this tab will go here…")
elif st.session_state.active_tab == "Call Tickets" and st.session_state.active_subtab == "Tickets":
    if st.session_state.pop('_show_ticket_saved_dialog', False):
        _ticket_saved_dialog()
    # Header row: only New Ticket button (single block to avoid extra vertical spacing on mobile)
    # Wrap the button so CSS can target it without :has or text matching
    if not _is_popup_view() and not st.session_state.get("_show_ticket_detail"):
        st.markdown('<div id="new-ticket-btn">', unsafe_allow_html=True)
        new_clicked = st.button("New Call Ticket", key="new_ticket_btn", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        new_clicked = False

    # New Ticket form
    if new_clicked:
        if _is_new_form_dirty():
            st.session_state['_confirm_new_ticket'] = True
            _confirm_new_ticket_dialog()
        else:
            _reset_new_ticket_fields()

    if st.session_state.get("_show_ticket_detail"):
        st.session_state["show_new_form"] = False

    if st.session_state.get("show_new_form", False):
        st.subheader("Add New Ticket")
    # initialize message keys if missing
        for _k in ['autofill_message_ai','autofill_level_ai','autofill_message_c','autofill_level_c','autofill_message_o','autofill_level_o']:
            if _k not in st.session_state:
                st.session_state[_k] = ''

        # Create tabs (always define before use); allow preselecting Complaints via deep link
        _base_tabs = ["Activities & Inquiries", "Complaints", "OSP Appointments"]
        _focus = st.session_state.get("_new_ticket_focus")
        if _is_popup_view():
            if _focus in _base_tabs:
                _tabs_order = [_focus]
            else:
                _tabs_order = ["Complaints"]
        else:
            _tabs_order = _base_tabs.copy()
            if _focus in _tabs_order:
                # rotate so focus tab is first
                i = _tabs_order.index(_focus)
                _tabs_order = _tabs_order[i:] + _tabs_order[:i]

        tabs_by_label = {}
        if len(_tabs_order) == 1:
            label = _tabs_order[0]
            tabs_by_label[label] = st.container()
        else:
            tabs = st.tabs(_tabs_order)
            tabs_by_label = {label: tabs[i] for i, label in enumerate(_tabs_order)}

        # ---- Activities & Inquiries ----
        if "Activities & Inquiries" in tabs_by_label:
            with tabs_by_label["Activities & Inquiries"]:
                a0c1, a0c2, a0c3 = st.columns(3)
                with a0c1:
                    st.selectbox("Activity / Inquiry Type", ACTIVITY_TYPES, index=None, placeholder="", key="sb_type_of_activity_inquiries_1")
                with a0c2: st.empty()
                with a0c3: st.empty()

                a1c1, a1c2, a1c3 = st.columns(3)
                with a1c1:
                    st.selectbox(
                        "Channel",
                        CHANNEL_OPTIONS,
                        index=(0 if CHANNEL_OPTIONS else None),
                        placeholder="",
                        key="channel_ai",
                    )
                with a1c2:
                    st.empty()
                with a1c3:
                    st.empty()

                with st.form("form_ai", clear_on_submit=False):
                    # Show AI toast just above first row (ONT ID)
                    _msg = st.session_state.get('autofill_message_ai')
                    _lvl = st.session_state.get('autofill_level_ai')
                    if _msg:
                        (st.warning if _lvl == 'warning' else st.info)(_msg)
                    if st.session_state.get("ai_pending_clear"):
                        st.session_state["ont_ai"] = ""
                        st.session_state["ont_ai_locked"] = ""
                        st.session_state["description_ai"] = ""
                        st.session_state["outage_start_ai"] = None
                        st.session_state["outage_end_ai"] = None
                        st.session_state["ai_pending_clear"] = False

                    ac1, ac2, ac3 = st.columns(3)
                    with ac1:
                        _ai_disabled = _is_popup_view()
                        st.text_input(
                            "ONT ID",
                            key="ont_ai",
                            placeholder="Enter ONT ID" if not _ai_disabled else "",
                            disabled=_ai_disabled,
                        )
                    with ac2:
                        call_type = st.selectbox("Call Type", CALL_TYPES, index=(0 if CALL_TYPES else None), placeholder="", key="sb_call_type_1")
                    with ac3:
                        # Created By is fixed to current user during creation
                        st.empty()

                    # Row 2: conditional iQ Digicare Issue Type
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        activity_choice = st.session_state.get("sb_type_of_activity_inquiries_1")
                        digicare_issue = ""
                        if activity_choice == "iQ Digicare":
                            digicare_issue = st.selectbox("iQ Digicare Issue Type", DIGICARE_ISSUES, index=None, placeholder="", key="sb_digicare_issue_1")
                    with b2:
                        st.empty()
                    with b3:
                        st.empty()

                    is_refund_info = activity_choice == "Refund Info"
                    is_refund_request = activity_choice == "Refund Request"

                    # Show outage fields only for Refund Request
                    if is_refund_request:
                        st.session_state.setdefault("outage_start_ai", None)
                        st.session_state.setdefault("outage_end_ai", None)
                        r_out1, r_out2, r_out3 = st.columns(3)
                        with r_out1:
                            outage_start_ai = st.date_input("Outage Start Date", key="outage_start_ai")
                        with r_out2:
                            outage_end_ai = st.date_input("Outage End Date", key="outage_end_ai")
                        with r_out3:
                            st.empty()
                    else:
                        st.session_state.pop("outage_start_ai", None)
                        st.session_state.pop("outage_end_ai", None)

                    # Description is needed for select activity types
                    show_desc = activity_choice in (
                        "Faulty Device & Adapter",
                        "Information",
                        "Refund Info",
                        "Refund Request",
                    )
                    if show_desc:
                        description = st.text_area(
                            "Description",
                            height=100,
                            placeholder="Enter details…",
                            key="description_ai",
                        )
                    else:
                        st.session_state["description_ai"] = ""
                        description = ""

                    save_ai = st.form_submit_button("Save Activities & Inquiries")
                    if save_ai:
                        activity_type_val = st.session_state.get("sb_type_of_activity_inquiries_1")
                        channel_ai_val = st.session_state.get("channel_ai") or ""
                        # Auto-assign creator to current user
                        created_by_ai = DEFAULT_CREATED_BY
                        missing = []
                        outage_start_val = st.session_state.get("outage_start_ai") if is_refund_request else None
                        outage_end_val = st.session_state.get("outage_end_ai") if is_refund_request else None
                        if not st.session_state.get("ont_ai", "").strip():
                            missing.append("ONT ID")
                        if not activity_type_val:
                            missing.append("Activity / Inquiry Type")
                        if not call_type:
                            missing.append("Call Type")
                        if not channel_ai_val:
                            missing.append("Channel")
                        # Require Description only for specific activity types
                        if (
                            activity_type_val in (
                                "Faulty Device & Adapter",
                                "Information",
                                "Refund Info",
                                "Refund Request",
                            )
                        ) and (not str(description).strip()):
                            missing.append("Description")
                        if is_refund_request:
                            if not outage_start_val:
                                missing.append("Outage Start Date")
                            if not outage_end_val:
                                missing.append("Outage End Date")
                        if missing:
                            _color_missing_labels(missing)
                            st.error("Please fill required fields: " + ", ".join(missing))
                        else:
                            outage_start_str = outage_start_val.isoformat() if outage_start_val else ""
                            outage_end_str = outage_end_val.isoformat() if outage_end_val else ""
                            add_ticket({
                                "ticket_group": "Activities & Inquiries",
                                "ont_id": st.session_state["ont_ai"],
                                "call_type": call_type,
                                "channel": channel_ai_val,
                                "description": description,
                                "activity_inquiry_type": activity_type_val,
                                "digicare_issue_type": st.session_state.get("sb_digicare_issue_1", "") if activity_type_val == "iQ Digicare" else "",
                                "outage_start_date": outage_start_str if is_refund_request else "",
                                "outage_end_date": outage_end_str if is_refund_request else "",
                                "created_by": created_by_ai,
                            })
                            st.success("Activities & Inquiries ticket added.")
                            st.session_state['_ticket_saved_message'] = "Activities & Inquiries ticket created successfully."
                            st.session_state['_show_ticket_saved_dialog'] = True
                            st.session_state.show_new_form = False
                            st.rerun()

                # Developer notes: render activities & inquiries markdown below the form inside an expander
                try:
                    _dev_md = load_implementation_note("activities_inquiries.md")
                    if _dev_md:
                        with st.expander("Implementation notes (Activities & Inquiries)", expanded=False):
                            st.markdown('<div class="implementation-notes-tag"></div>', unsafe_allow_html=True)
                            st.markdown(_dev_md)
                except Exception:
                    pass

        # ---- Complaints ----
        if "Complaints" in tabs_by_label:
            with tabs_by_label["Complaints"]:
                r0c1, r0c2, r0c3 = st.columns(3)
                with r0c1:
                    st.selectbox("Complaint Type", COMPLAINT_TYPES, index=None, placeholder="", key="sb_type_of_complaint_1")
                with r0c2:
                    # Refund Type appears only when Complaint Type is Refund
                    if st.session_state.get("sb_type_of_complaint_1") == "Refund":
                        st.selectbox("Refund Type", REFUND_TYPES, index=None, placeholder="", key="sb_refund_type_1")
                    else:
                        st.session_state["sb_refund_type_1"] = ""
                        st.empty()
                with r0c3:
                    st.empty()

                # Clear outage dates unless Refund Type is explicitly 'Refund Request'
                _ct_norm = str(st.session_state.get("sb_type_of_complaint_1") or "").strip().lower()
                _rt_norm = str(st.session_state.get("sb_refund_type_1") or "").strip().lower()
                if not (_ct_norm == "refund" and _rt_norm == "refund request"):
                    # Remove invalid defaults so date_input can render cleanly when shown
                    st.session_state.pop("outage_start_c", None)
                    st.session_state.pop("outage_end_c", None)

                # Place Complaint Status on a new row below (first column), still outside the form to allow reruns on change
                s0c1, s0c2, s0c3 = st.columns(3)
                with s0c1:
                    st.selectbox("Complaint Status", COMP_STATUS, index=None, placeholder="", key="sb_complaint_status_1")
                with s0c2:
                    st.empty()
                with s0c3:
                    st.empty()

                ch0c1, ch0c2, ch0c3 = st.columns(3)
                with ch0c1:
                    st.selectbox(
                        "Channel",
                        CHANNEL_OPTIONS,
                        index=(0 if CHANNEL_OPTIONS else None),
                        placeholder="",
                        key="channel_c",
                    )
                with ch0c2:
                    st.empty()
                with ch0c3:
                    st.empty()

                with st.form("form_complaint", clear_on_submit=False):
                    # Show Complaints toast just above first row (ONT ID)
                    _msg = st.session_state.get('autofill_message_c')
                    _lvl = st.session_state.get('autofill_level_c')
                    if _msg:
                        (st.warning if _lvl == 'warning' else st.info)(_msg)
                    if st.session_state.get("c_pending_clear"):
                        st.session_state["ont_c"] = ""
                        st.session_state["olt_c"] = ""
                        st.session_state["olt_c_filled"] = ""
                        st.session_state["ont_c_locked"] = ""
                        st.session_state["ont_model"] = ""
                        st.session_state["kurdtel_status_c"] = ""
                        st.session_state["ip_c"] = ""
                        st.session_state["vlan_c"] = ""
                        st.session_state["packet_loss_c"] = ""
                        st.session_state["high_ping_c"] = ""
                        st.session_state["ont_power_level_c"] = ""
                        st.session_state["olt_power_level_c"] = ""
                        st.session_state["sip_c_value"] = ""
                        st.session_state["name_c_value"] = ""
                        st.session_state["kurdtel_device_type_c_value"] = ""
                        st.session_state["issue_type_kurdtel"] = ""
                        st.session_state["second_number_c"] = ""
                        st.session_state["description_c"] = ""
                        st.session_state["visit_required_c"] = False
                        st.session_state["c_pending_clear"] = False

                    r1c1, r1c2, r1c3 = st.columns(3)
                    with r1c1:
                        locked_c = bool(st.session_state.get("ont_c_locked"))
                        c_autofill = False
                        c_remove = None
                        if locked_c and _is_popup_view():
                            st.text_input("ONT ID", key="ont_c", placeholder="Enter ONT ID", disabled=True)
                        else:
                            sub_l, sub_r = st.columns([4,1])
                            with sub_l:
                                st.text_input("ONT ID", key="ont_c", placeholder="Enter ONT ID", disabled=locked_c)
                            with sub_r:
                                st.markdown("<div style='height:1.9rem;'></div>", unsafe_allow_html=True)
                                if not locked_c:
                                    c_autofill = st.form_submit_button("🔍︎", help="Fetch and autofill related details", use_container_width=True)
                                else:
                                    if not _is_popup_view():
                                        c_remove = st.form_submit_button("❌︎", use_container_width=True)
                                    else:
                                        st.empty()
                    # Handle deferred deep-link autofill once
                    if st.session_state.pop("_do_autofill_c", False):
                        if st.session_state.get("ont_c", "").strip():
                            st.session_state["olt_c"] = "NTWK-Sul-Pasha-OLT-00"
                            try:
                                # Prefer an option starting with "844"; fallback to 3rd option, else 1st
                                _ont_target = next((m for m in (ONT_MODELS or []) if str(m).strip().startswith("844")), None)
                                if _ont_target is None and len(ONT_MODELS or []) >= 3:
                                    _ont_target = ONT_MODELS[2]
                                if _ont_target is None:
                                    _ont_target = ONT_MODELS[0] if ONT_MODELS else ""
                                st.session_state["ont_model"] = _ont_target
                            except Exception:
                                st.session_state["ont_model"] = st.session_state.get("ont_model", "")
                            # Autofill Kurdtel Service Status on deep-link when relevant
                            try:
                                if st.session_state.get("sb_type_of_complaint_1") == "Kurdtel":
                                    st.session_state["kurdtel_status_c"] = KURDTEL_SERVICE_STATUS[0] if KURDTEL_SERVICE_STATUS else ""
                            except Exception:
                                st.session_state["kurdtel_status_c"] = st.session_state.get("kurdtel_status_c", "")
                            # Autofill IP/VLAN on deep-link autofill
                            st.session_state["ip_c"] = "10.49.72.000"
                            st.session_state["vlan_c"] = "3021"
                            st.session_state["ont_power_level_c"] = "-21.000"
                            st.session_state["olt_power_level_c"] = "-23.700"
                            if st.session_state.get("sb_type_of_complaint_1") == "Kurdtel":
                                st.session_state["sip_c_value"] = "12345"
                                st.session_state["name_c_value"] = "Test"
                                st.session_state["kurdtel_device_type_c_value"] = "Test"
                            st.session_state["olt_c_filled"] = True
                            st.session_state["ont_c_locked"] = True
                            st.session_state["autofill_message_c"] = "Fields autofilled."
                            st.session_state["autofill_level_c"] = "info"
                            st.rerun()
                    # Row 1: c2 = Call Type, c3 varies by complaint type
                    with r1c2:
                        call_type_c = st.selectbox("Call Type", CALL_TYPES, index=(0 if CALL_TYPES else None), placeholder="", key="sb_call_type_2")
                    is_kurdtel = st.session_state.get("sb_type_of_complaint_1") == "Kurdtel"
                    employee_suggestion = st.session_state.get("sb_employee_suggestion_1", "")
                    with r1c3:
                        if is_kurdtel:
                            if not employee_suggestion and EMP_SUGGESTION:
                                employee_suggestion = EMP_SUGGESTION[0]
                            st.session_state["sb_employee_suggestion_1"] = employee_suggestion or ""
                            st.selectbox(
                                "Issue Type",
                                ISSUE_TYPES, index=None, placeholder="",
                                key="issue_type_kurdtel"
                            )
                        else:
                            employee_suggestion = st.selectbox("Employee Suggestion", EMP_SUGGESTION, index=None, placeholder="", key="sb_employee_suggestion_1")

                    packet_loss_val = st.session_state.get("packet_loss_c", "")
                    high_ping_val = st.session_state.get("high_ping_c", "")
                    second_number_value = st.session_state.get("second_number_c", "")
                    olt_c_val = st.session_state.get("olt_c", "")
                    if "ont_power_level_c" not in st.session_state:
                        st.session_state["ont_power_level_c"] = ""
                    if "olt_power_level_c" not in st.session_state:
                        st.session_state["olt_power_level_c"] = ""
                    if not locked_c:
                        if c_autofill:
                            if st.session_state.get("ont_c", "").strip():
                                st.session_state["olt_c"] = "NTWK-Sul-Pasha-OLT-00"
                                try:
                                    # Prefer an option starting with "844"; fallback to 3rd option, else 1st
                                    _ont_target = next((m for m in (ONT_MODELS or []) if str(m).strip().startswith("844")), None)
                                    if _ont_target is None and len(ONT_MODELS or []) >= 3:
                                        _ont_target = ONT_MODELS[2]
                                    if _ont_target is None:
                                        _ont_target = ONT_MODELS[0] if ONT_MODELS else ""
                                    st.session_state["ont_model"] = _ont_target
                                except Exception:
                                    st.session_state["ont_model"] = ""
                                # Autofill IP/VLAN on manual autofill
                                st.session_state["ip_c"] = "10.49.72.000"
                                st.session_state["vlan_c"] = "3021"
                                st.session_state["ont_power_level_c"] = "-21.000"
                                st.session_state["olt_power_level_c"] = "-23.700"
                                if st.session_state.get("sb_type_of_complaint_1") == "Kurdtel":
                                    st.session_state["sip_c_value"] = "12345"
                                    st.session_state["name_c_value"] = "Test"
                                    st.session_state["kurdtel_device_type_c_value"] = "Test"
                                st.session_state["olt_c_filled"] = True
                                st.session_state["ont_c_locked"] = True
                                try:
                                    if st.session_state.get("sb_type_of_complaint_1") == "Kurdtel":
                                        st.session_state["kurdtel_status_c"] = KURDTEL_SERVICE_STATUS[0] if KURDTEL_SERVICE_STATUS else ""
                                except Exception:
                                    st.session_state["kurdtel_status_c"] = st.session_state.get("kurdtel_status_c", "")
                                st.session_state["autofill_message_c"] = "Fields autofilled."
                                st.session_state["autofill_level_c"] = "info"
                                st.rerun()
                            else:
                                st.session_state["autofill_message_c"] = "Enter ONT ID first."
                                st.session_state["autofill_level_c"] = "warning"
                                st.rerun()
                    else:
                        if c_remove:
                            st.session_state["c_pending_clear"] = True
                            st.session_state["autofill_message_c"] = "Cleared autofill."
                            st.session_state["autofill_level_c"] = "info"
                            st.rerun()

                    # Complaint Status is controlled above the form
                    # (r0c2 outside the form)

                    _status_choice = st.session_state.get("sb_complaint_status_1")
                    _status_norm = (str(_status_choice).strip().lower() if _status_choice else "")
                    _ct_choice = st.session_state.get("sb_type_of_complaint_1")
                    _ct_norm = (str(_ct_choice).strip().lower() if _ct_choice else "")
                    _show_second = bool((_status_norm in {"not solved", "pending"}) or (_ct_norm == "problem arising from the extender"))

                    r2c1, r2c2, r2c3 = st.columns(3)
                    if is_kurdtel:
                        with r2c1:
                            if _show_second:
                                second_number_value = st.text_input(
                                    "Second Number",
                                    key="second_number_c",
                                    placeholder="Click 🔍︎ to auto fill",
                                )
                            else:
                                st.empty()
                                second_number_value = ""
                                st.session_state["second_number_c"] = ""
                        with r2c2:
                            st.empty()
                        with r2c3:
                            st.empty()

                        r3c1, r3c2, r3c3 = st.columns(3)
                        with r3c1:
                            st.session_state["name_c_display"] = st.session_state.get("name_c_value") or ""
                            st.text_input(
                                "Name",
                                key="name_c_display",
                                disabled=True,
                                placeholder="Click 🔍︎ to auto fill",
                            )
                        with r3c2:
                            st.selectbox(
                                "Kurdtel Service Status",
                                KURDTEL_SERVICE_STATUS, index=None, placeholder="Click 🔍︎ to auto fill",
                                key="kurdtel_status_c", disabled=True
                            )
                        with r3c3:
                            st.session_state["kurdtel_device_type_c_display"] = st.session_state.get("kurdtel_device_type_c_value") or ""
                            st.text_input(
                                "Kurdtel Device Type",
                                key="kurdtel_device_type_c_display",
                                disabled=True,
                                placeholder="Click 🔍︎ to auto fill",
                            )
                        root_cause = st.session_state.get("sb_root_cause_1", "")
                        device_location = st.session_state.get("sb_device_location_1", "")
                        ont_model = st.session_state.get("ont_model", "")
                    else:
                        with r2c1:
                            root_cause = st.selectbox("Root Cause", ROOT_CAUSE, index=None, placeholder="", key="sb_root_cause_1")
                        with r2c2:
                            device_location = st.selectbox("Device Location", DEVICE_LOC, index=None, placeholder="", key="sb_device_location_1")
                        with r2c3:
                            if _show_second:
                                second_number_value = st.text_input("Second Number", key="second_number_c")
                            else:
                                st.empty()
                                second_number_value = ""
                                st.session_state["second_number_c"] = ""

                    outage_start = None
                    outage_end = None
                    if is_kurdtel:
                        packet_loss_val = st.session_state.get("packet_loss_c", "")
                        high_ping_val = st.session_state.get("high_ping_c", "")
                        st.session_state.setdefault("ip_c", st.session_state.get("ip_c", ""))
                        st.session_state.setdefault("vlan_c", st.session_state.get("vlan_c", ""))
                        olt_c_val = st.session_state.get("olt_c", "")
                    else:
                        # r3 (new order): Packet Loss, High Ping, spacer
                        r3c1, r3c2, r3c3 = st.columns(3)
                        with r3c1:
                            packet_loss_val = st.text_input("Packet Loss (%)", key="packet_loss_c", placeholder="e.g., 10")
                        with r3c2:
                            high_ping_val = st.text_input("High Ping (ms)", key="high_ping_c", placeholder="e.g., 180")
                        with r3c3:
                            st.empty()

                        # r4: Online Game / Refund Request controls
                        r4c1, r4c2, r4c3 = st.columns(3)
                        with r4c1:
                            if st.session_state.get("sb_type_of_complaint_1") == "Online Game Issue":
                                # Online Game with 'Other' + adjacent input that shows instantly (no server rerun needed)
                                _og_options = list(ONLINE_GAMES or [])
                                if "Other" not in _og_options:
                                    _og_options.append("Other")
                                st.selectbox(
                                    "Online Game",
                                    _og_options, index=None, placeholder="",
                                    key="online_game_c"
                                )
                                # Pure CSS toggle using :has() on stable Streamlit keys (no reruns, no JS)
                                st.markdown(
                                    """
                                    <style>
                                    /* Hide the Other input by default */
                                    .st-key-online_game_other_c { display: none !important; }
                                    /* Show globally when Online Game is set to Other (works across columns) */
                                    .stApp:has(.st-key-online_game_c div[value=\"Other\"]) .st-key-online_game_other_c { display: block !important; }
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            elif (
                                str(st.session_state.get("sb_type_of_complaint_1") or "").strip().lower() == "refund"
                            ) and (
                                str(st.session_state.get("sb_refund_type_1") or "").strip().lower() == "refund request"
                            ):
                                # Ensure prior clears didn't leave an invalid default (e.g., empty string)
                                if isinstance(st.session_state.get("outage_start_c", None), str):
                                    st.session_state.pop("outage_start_c", None)
                                outage_start = st.date_input("Outage Start Date", key="outage_start_c")
                            else:
                                st.empty()
                        with r4c2:
                            if st.session_state.get("sb_type_of_complaint_1") == "Online Game Issue":
                                st.text_input("Other (Online Game)", key="online_game_other_c", placeholder="Enter game title")
                            elif (
                                str(st.session_state.get("sb_type_of_complaint_1") or "").strip().lower() == "refund"
                            ) and (
                                str(st.session_state.get("sb_refund_type_1") or "").strip().lower() == "refund request"
                            ):
                                if isinstance(st.session_state.get("outage_end_c", None), str):
                                    st.session_state.pop("outage_end_c", None)
                                outage_end = st.date_input("Outage End Date", key="outage_end_c")
                            else:
                                st.empty()
                        with r4c3:
                            st.empty()

                        # r5: ONT Model, ONT Power Level, OLT Power Level
                        r5c1, r5c2, r5c3 = st.columns(3)
                        with r5c1:
                            ont_model = st.selectbox("ONT Model", ONT_MODELS, index=None, placeholder="Click 🔍︎ to auto fill", key="ont_model")
                        with r5c2:
                            st.text_input(
                                "ONT Power Level",
                                key="ont_power_level_c",
                                placeholder="Click 🔍︎ to auto fill",
                                disabled=True,
                            )
                        with r5c3:
                            st.text_input(
                                "OLT Power Level",
                                key="olt_power_level_c",
                                placeholder="Click 🔍︎ to auto fill",
                                disabled=True,
                            )

                        # r6: OLT, IP (disabled), VLAN (disabled)
                        r6c1, r6c2, r6c3 = st.columns(3)
                        with r6c1:
                            olt_c_val = st.text_input("OLT", key="olt_c", placeholder="Click 🔍︎ to auto fill")
                        with r6c2:
                            st.text_input("IP", key="ip_c", placeholder="Click 🔍︎ to auto fill", disabled=True)
                        with r6c3:
                            st.text_input("VLAN", key="vlan_c", placeholder="Click 🔍︎ to auto fill", disabled=True)
                    # Disabled Call-Back / Follow-Up controls during creation
                    r_cb1, r_cb2, r_cb3 = st.columns(3)
                    with r_cb1:
                        st.selectbox(
                            "Call-Back Status", CALLBACK_STATUS,
                            index=None, placeholder="", key="callback_status_c", disabled=True
                        )
                    with r_cb2:
                        st.selectbox(
                            "Call-Back Reason", CALLBACK_REASON,
                            index=None, placeholder="", key="callback_reason_c", disabled=True
                        )
                    with r_cb3:
                        st.selectbox(
                            "Follow-Up Status", FOLLOWUP_STATUS,
                            index=None, placeholder="", key="followup_status_c", disabled=True
                        )

                    if is_kurdtel:
                        sip_c1, sip_c2, sip_c3 = st.columns(3)
                        with sip_c1:
                            st.session_state["sip_c_display"] = st.session_state.get("sip_c_value") or ""
                            st.text_input(
                                "SIP",
                                key="sip_c_display",
                                disabled=True,
                                placeholder="Click 🔍︎ to auto fill",
                            )
                        with sip_c2:
                            st.empty()
                        with sip_c3:
                            st.empty()

                    description_c = st.text_area(
                        "Description",
                        height=100,
                        placeholder="Describe the complaint…",
                        key="description_c",
                    )

                    # r7: Reminder toggle (Created By auto-assigned)
                    if "reminder_date_c" not in st.session_state:
                        st.session_state["reminder_date_c"] = None
                    if "reminder_time_c" not in st.session_state:
                        st.session_state["reminder_time_c"] = None
                    if "reminder_enabled_c" not in st.session_state:
                        st.session_state["reminder_enabled_c"] = False
                    if "reminder_recipient_c" not in st.session_state:
                        st.session_state["reminder_recipient_c"] = []
                    if "reminder_note_c" not in st.session_state:
                        st.session_state["reminder_note_c"] = ""
                    if "visit_required_c" not in st.session_state:
                        st.session_state["visit_required_c"] = False

                    r7c1, r7c2 = st.columns(2)
                    _current_recipients = st.session_state.get("reminder_recipient_c", [])
                    if not isinstance(_current_recipients, list):
                        _current_recipients = _parse_reminder_recipients(_current_recipients)
                    if not _current_recipients and DEFAULT_CREATED_BY:
                        _current_recipients = [DEFAULT_CREATED_BY]
                    _current_recipients = _normalize_reminder_selection(_current_recipients)
                    st.session_state["reminder_recipient_c"] = _current_recipients
                    with r7c1:
                        reminder_enabled = st.checkbox("Set Reminder", key="reminder_enabled_c")
                    with r7c2:
                        st.empty()

                    st.markdown(
                        """
                        <style>
                        .st-key-reminder_recipient_c,
                        .st-key-reminder_date_c,
                        .st-key-reminder_time_c,
                        .st-key-reminder_note_c {
                            display: none !important;
                        }
                        .stApp:has(.st-key-reminder_enabled_c input[type=\"checkbox\"]:checked) .st-key-reminder_recipient_c,
                        .stApp:has(.st-key-reminder_enabled_c input[type=\"checkbox\"]:checked) .st-key-reminder_date_c,
                        .stApp:has(.st-key-reminder_enabled_c input[type=\"checkbox\"]:checked) .st-key-reminder_time_c,
                        .stApp:has(.st-key-reminder_enabled_c input[type=\"checkbox\"]:checked) .st-key-reminder_note_c {
                            display: block !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    rem_r1c1, rem_r1c2, rem_r1c3 = st.columns(3)
                    with rem_r1c1:
                        _recipient_options = _ensure_recipient_options(
                            REMINDER_RECIPIENT_OPTIONS,
                            st.session_state.get("reminder_recipient_c", []),
                        )
                        _selected_recipients_current = st.multiselect(
                            "Reminder Recipient",
                            _recipient_options,
                            key="reminder_recipient_c",
                            placeholder="Select recipient(s)…",
                        )
                        _selected_recipients_current = _normalize_reminder_selection(_selected_recipients_current)
                    with rem_r1c2:
                        st.date_input("Reminder Date", key="reminder_date_c")
                    with rem_r1c3:
                        if st.session_state.get("reminder_date_c") and not st.session_state.get("reminder_time_c"):
                            try:
                                st.session_state["reminder_time_c"] = _now_local().time().replace(second=0, microsecond=0)
                            except Exception:
                                st.session_state["reminder_time_c"] = None
                        st.time_input("Reminder Time", key="reminder_time_c")
                    st.text_area(
                        "Reminder Note",
                        key="reminder_note_c",
                        height=80,
                        placeholder="Add reminder details…",
                    )

                    st.checkbox("Visit Required", key="visit_required_c")

                    save_c = st.form_submit_button("Save Complaint")
                    if save_c:
                        missing = []
                        packet_loss_clean = str(st.session_state.get("packet_loss_c") or "").strip()
                        high_ping_clean = str(st.session_state.get("high_ping_c") or "").strip()
                        invalid_labels = []
                        invalid_messages = []

                        ont_val = st.session_state.get("ont_c", "").strip()
                        ct_val = st.session_state.get("sb_type_of_complaint_1")
                        es_val = employee_suggestion if "employee_suggestion" in locals() else None
                        rc_val = root_cause if "root_cause" in locals() else None
                        om_val = ont_model if "ont_model" in locals() else None
                        dl_val = device_location if "device_location" in locals() else None
                        cs_val = st.session_state.get("sb_complaint_status_1")
                        call_type_val = call_type_c if "call_type_c" in locals() else None
                        olt_val = st.session_state.get("olt_c", "").strip()
                        sn_val = str(second_number_value).strip()
                        desc_val = str(description_c).strip() if "description_c" in locals() else ""
                        ks_val = st.session_state.get("kurdtel_status_c", "").strip()
                        online_game_val = str(st.session_state.get("online_game_c") or "").strip()
                        online_game_other = str(st.session_state.get("online_game_other_c") or "").strip()
                        refund_type_val = str(st.session_state.get("sb_refund_type_1") or "").strip()
                        created_by_val = DEFAULT_CREATED_BY
                        reminder_enabled = bool(st.session_state.get("reminder_enabled_c"))
                        channel_c_val = st.session_state.get("channel_c") or ""
                        reminder_recipients_selection = st.session_state.get("reminder_recipient_c")
                        if not isinstance(reminder_recipients_selection, list):
                            reminder_recipients_selection = _parse_reminder_recipients(reminder_recipients_selection)
                        reminder_recipients_selection = _normalize_reminder_selection(reminder_recipients_selection)
                        reminder_recipient = _serialize_reminder_recipients(reminder_recipients_selection)
                        reminder_note = str(st.session_state.get("reminder_note_c") or "").strip()
                        rem_date = st.session_state.get("reminder_date_c")
                        rem_time = st.session_state.get("reminder_time_c")
                        reminder_at_str = ""
                        visit_required_flag = bool(st.session_state.get("visit_required_c"))

                        if packet_loss_clean:
                            try:
                                packet_loss_num = float(packet_loss_clean)
                            except ValueError:
                                if "Packet Loss (%)" not in invalid_labels:
                                    invalid_labels.append("Packet Loss (%)")
                                invalid_messages.append("Packet Loss (%) must be a number between 1 and 100")
                            else:
                                if not (1 <= packet_loss_num <= 100):
                                    if "Packet Loss (%)" not in invalid_labels:
                                        invalid_labels.append("Packet Loss (%)")
                                    invalid_messages.append("Packet Loss (%) must be between 1 and 100")
                                else:
                                    normalized_packet = f"{packet_loss_num:g}"
                                    if normalized_packet != packet_loss_clean:
                                        st.session_state["packet_loss_c"] = normalized_packet
                                        packet_loss_clean = normalized_packet

                        if high_ping_clean:
                            try:
                                high_ping_num = float(high_ping_clean)
                            except ValueError:
                                if "High Ping (ms)" not in invalid_labels:
                                    invalid_labels.append("High Ping (ms)")
                                invalid_messages.append("High Ping (ms) must be a number")
                            else:
                                normalized_high = f"{high_ping_num:g}"
                                if normalized_high != high_ping_clean:
                                    st.session_state["high_ping_c"] = normalized_high
                                    high_ping_clean = normalized_high

                        if not ct_val: missing.append("Complaint Type")
                        if not ont_val: missing.append("ONT ID")
                        if not es_val: missing.append("Employee Suggestion")
                        if ct_val != "Kurdtel" and not rc_val: missing.append("Root Cause")
                        if not om_val: missing.append("ONT Model")
                        if ct_val != "Kurdtel" and not dl_val: missing.append("Device Location")
                        if not cs_val: missing.append("Complaint Status")
                        if not call_type_val: missing.append("Call Type")
                        if not olt_val: missing.append("OLT")
                        if not channel_c_val: missing.append("Channel")
                        # Require Second Number only when visible/required (normalize for robustness)
                        _cs_norm = (str(cs_val).strip().lower() if cs_val else "")
                        _ct_norm2 = (str(ct_val).strip().lower() if ct_val else "")
                        _require_second = bool((_cs_norm in {"not solved", "pending"}) or (_ct_norm2 == "problem arising from the extender"))
                        if _require_second and not sn_val:
                            missing.append("Second Number")
                        if not desc_val: missing.append("Description")
                        if ct_val == "Kurdtel" and not ks_val:
                            missing.append("Kurdtel Service Status")
                        if ct_val == "Refund" and not refund_type_val:
                            missing.append("Refund Type")

                        # Require Online Game selection when complaint type is Online Game Issue
                        if ct_val == "Online Game Issue" and not online_game_val:
                            missing.append("Online Game")

                        # Extra validation for Online Game 'Other'
                        if ct_val == "Online Game Issue" and online_game_val == "Other" and not online_game_other:
                            missing.append("Other (Online Game)")

                        # Validate reminder when enabled
                        if reminder_enabled:
                            if not reminder_recipients_selection:
                                missing.append("Reminder Recipient")
                            if not rem_date:
                                missing.append("Reminder Date")
                            if not rem_time:
                                missing.append("Reminder Time")
                            if rem_date and rem_time:
                                try:
                                    _combined_dt = datetime.combine(rem_date, rem_time)
                                    if _combined_dt <= _now_local():
                                        missing.append("Reminder Date/Time (future)")
                                    else:
                                        reminder_at_str = _combined_dt.isoformat()
                                except Exception:
                                    missing.append("Reminder Date/Time")

                        error_messages = []
                        if missing:
                            _color_missing_labels(missing)
                            error_messages.append("Please fill in all required fields: " + ", ".join(missing))
                        if invalid_messages:
                            _color_missing_labels(invalid_labels)
                            error_messages.append("Please fix the following: " + "; ".join(invalid_messages))

                        if error_messages:
                            st.error(" ".join(error_messages))
                        else:
                            add_ticket({
                                "ticket_group": "Complaints",
                                "ont_id": ont_val,
                                "call_type": call_type_val,
                                "description": desc_val,
                                "complaint_type": ct_val,
                                "refund_type": refund_type_val if ct_val == "Refund" else "",
                                "channel": channel_c_val,
                                "employee_suggestion": es_val,
                                "device_location": (dl_val if ct_val != "Kurdtel" else ""),
                                "root_cause": rc_val,
                                "ont_model": om_val,
                                "complaint_status": cs_val,
                                "kurdtel_service_status": ks_val if ct_val == "Kurdtel" else "",
                                "online_game": ((online_game_val == "Other") and online_game_other or online_game_val) if ct_val == "Online Game Issue" else "",
                                "olt": olt_val,
                                "second_number": sn_val,
                                "created_by": created_by_val,
                                "outage_start_date": (outage_start.isoformat() if hasattr(outage_start, 'isoformat') else (str(outage_start) if outage_start else "")),
                                "outage_end_date": (outage_end.isoformat() if hasattr(outage_end, 'isoformat') else (str(outage_end) if outage_end else "")),
                                "ip": st.session_state.get("ip_c", ""),
                                "vlan": st.session_state.get("vlan_c", ""),
                                "packet_loss": packet_loss_clean,
                                "high_ping": high_ping_clean,
                                "callback_status": (st.session_state.get("callback_status_c") or ""),
                                "callback_reason": (st.session_state.get("callback_reason_c") or ""),
                                "followup_status": (st.session_state.get("followup_status_c") or ""),
                                "visit_required": visit_required_flag,
                                "reminder_enabled": reminder_enabled,
                                "reminder_recipient": reminder_recipient if reminder_enabled else "",
                                "reminder_note": reminder_note if reminder_enabled else "",
                                "reminder_at": reminder_at_str
                            })
                            st.success("Complaint ticket added.")
                            st.session_state['_ticket_saved_message'] = "Complaint ticket created successfully."
                            st.session_state['_show_ticket_saved_dialog'] = True
                            st.session_state.show_new_form = False
                            st.rerun()
                # Developer notes: render complaints markdown below the form inside an expander
                try:
                    _dev_md_c = load_implementation_note("complaints.md")
                    if _dev_md_c:
                        with st.expander("Implementation notes (Complaints)", expanded=False):
                            st.markdown('<div class="implementation-notes-tag"></div>', unsafe_allow_html=True)
                            st.markdown(_dev_md_c)
                except Exception:
                    pass

    # ---- OSP Appointments ----
    if 'tabs_by_label' in locals() and "OSP Appointments" in tabs_by_label:
        with tabs_by_label["OSP Appointments"]:
            o0c1, o0c2, o0c3 = st.columns(3)
            with o0c1:
                st.selectbox(
                    "OSP Appointment Type",
                    OSP_TYPES, index=None, placeholder="",
                    key="sb_osp_appointment_type_1"
                )
            with o0c2: st.empty()
            with o0c3: st.empty()

            oc0c1, oc0c2, oc0c3 = st.columns(3)
            with oc0c1:
                st.selectbox(
                    "Channel",
                    CHANNEL_OPTIONS,
                    index=(0 if CHANNEL_OPTIONS else None),
                    placeholder="",
                    key="channel_o",
                )
            with oc0c2:
                st.empty()
            with oc0c3:
                st.empty()

            with st.form("form_osp", clear_on_submit=False):
                # Show OSP toast just above first row (ONT ID)
                _msg = st.session_state.get('autofill_message_o')
                _lvl = st.session_state.get('autofill_level_o')
                if _msg:
                    (st.warning if _lvl == 'warning' else st.info)(_msg)
                if st.session_state.get("osp_pending_clear"):
                    st.session_state["ont_o"] = ""
                    st.session_state["ont_o_locked"] = ""
                    st.session_state["city_o"] = ""
                    st.session_state["fttg_o"] = ""
                    st.session_state["address_o"] = ""
                    st.session_state["olt_o"] = ""
                    st.session_state["line_card_o"] = ""
                    st.session_state["gpon_o"] = ""
                    st.session_state["issue_type_o"] = ""
                    st.session_state["second_number_o"] = ""
                    st.session_state["description_o"] = ""
                    st.session_state["osp_pending_clear"] = False

                or1c1, or1c2, or1c3 = st.columns(3)
                with or1c1:
                    locked_o = bool(st.session_state.get("ont_o_locked"))
                    o_autofill = False
                    o_remove = None
                    if locked_o and _is_popup_view():
                        st.text_input("ONT ID", key="ont_o", placeholder="Enter ONT ID", disabled=True)
                    else:
                        sub_l, sub_r = st.columns([4,1])
                        with sub_l:
                            st.text_input("ONT ID", key="ont_o", placeholder="Enter ONT ID", disabled=locked_o)
                        with sub_r:
                            st.markdown("<div style='height:1.9rem;'></div>", unsafe_allow_html=True)
                            if not locked_o:
                                o_autofill = st.form_submit_button("🔍︎", help="Fetch and autofill related details", use_container_width=True)
                            else:
                                if not _is_popup_view():
                                    o_remove = st.form_submit_button("❌︎", use_container_width=True)
                                else:
                                    st.empty()
                # Handle deferred deep-link autofill once for OSP
                if st.session_state.pop("_do_autofill_o", False):
                    if st.session_state.get("ont_o", "").strip():
                        try:
                            st.session_state["city_o"] = CITY_OPTIONS[0] if CITY_OPTIONS else ""
                        except Exception:
                            st.session_state["city_o"] = st.session_state.get("city_o", "")
                        st.session_state["fttg_o"] = "No"
                        st.session_state["address_o"] = "Rizgary Quarter 412, Building 64, Sulaymaniyah, Kurdistan Region"
                        st.session_state["olt_o"] = "NTWK-Sul-Pasha-OLT-00"
                        st.session_state["line_card_o"] = "Cisco NCS 5500"
                        st.session_state["gpon_o"] = "2.5G/1.25G"
                        st.session_state["ont_o_locked"] = True
                        st.session_state["autofill_message_o"] = "Fields autofilled."
                        st.session_state["autofill_level_o"] = "info"
                        st.rerun()

                with or1c2:
                    call_type_o = st.selectbox("Call Type", CALL_TYPES, index=(0 if CALL_TYPES else None), placeholder="", key="sb_call_type_3")
                with or1c3:
                    second_number_o = st.text_input("Second Number", key="second_number_o")

                if not locked_o:
                    if o_autofill:
                        if st.session_state.get("ont_o", "").strip():
                            try:
                                st.session_state["city_o"] = CITY_OPTIONS[0] if CITY_OPTIONS else ""
                            except Exception:
                                st.session_state["city_o"] = ""
                            st.session_state["fttg_o"] = "No"
                            st.session_state["address_o"] = "Rizgary Quarter 412, Building 64, Sulaymaniyah, Kurdistan Region"
                            st.session_state["olt_o"] = "NTWK-Sul-Pasha-OLT-00"
                            st.session_state["line_card_o"] = "Cisco NCS 5500"
                            st.session_state["gpon_o"] = "2.5G/1.25G"
                            st.session_state["ont_o_locked"] = True
                            st.session_state["autofill_message_o"] = "Fields autofilled."
                            st.session_state["autofill_level_o"] = "info"
                            st.rerun()
                        else:
                            st.session_state["autofill_message_o"] = "Enter ONT ID first."
                            st.session_state["autofill_level_o"] = "warning"
                            st.rerun()
                else:
                    if o_remove:
                        st.session_state["osp_pending_clear"] = True
                        st.session_state["autofill_message_o"] = "Cleared autofill."
                        st.session_state["autofill_level_o"] = "info"
                        st.rerun()

                # Row 2: Issue Type, FTTG, City
                or2c1, or2c2, or2c3 = st.columns(3)
                with or2c1:
                    issue_type_o = st.selectbox(
                        "Issue Type",
                        ISSUE_TYPES, index=None, placeholder="",
                        key="issue_type_o"
                    )
                with or2c2:
                    fttg_val = st.selectbox(
                        "FTTG",
                        FTTG_OPTIONS, index=None, placeholder="Click 🔍︎ to auto fill",
                        key="fttg_o", disabled=True
                    )
                with or2c3:
                    city_val = st.selectbox(
                        "City",
                        CITY_OPTIONS, index=None, placeholder="Click 🔍︎ to auto fill",
                        key="city_o", disabled=True
                    )

                # Row 3: OLT, Line Card, GPON
                or3c1, or3c2, or3c3 = st.columns(3)
                with or3c1:
                    st.text_input(
                        "OLT",
                        key="olt_o",
                        placeholder="Click 🔍︎ to auto fill",
                        disabled=True
                    )
                with or3c2:
                    st.text_input(
                        "Line Card",
                        key="line_card_o",
                        placeholder="Click 🔍︎ to auto fill",
                        disabled=True
                    )
                with or3c3:
                    st.text_input(
                        "GPON",
                        key="gpon_o",
                        placeholder="Click 🔍︎ to auto fill",
                        disabled=True
                    )

                # Row 4: Address (full width)
                address_o = st.text_area("Address", height=80, placeholder="Click 🔍︎ to auto fill", key="address_o")

                # Row 5: Description (full width)
                description_o = st.text_area(
                    "Description",
                    height=100,
                    placeholder="Describe the appointment…",
                    key="description_o",
                )

                submitted_o = st.form_submit_button("Save OSP Appointment")
                if submitted_o:
                    missing = []

                    osp_val   = st.session_state.get("sb_osp_appointment_type_1")
                    ont_val   = st.session_state.get("ont_o", "").strip()
                    city_sel  = st.session_state.get("city_o", "").strip()
                    sn_val    = str(second_number_o).strip() if "second_number_o" in locals() else ""
                    issue_sel = issue_type_o if "issue_type_o" in locals() else None
                    call_sel  = call_type_o if "call_type_o" in locals() else None
                    fttg_sel  = st.session_state.get("fttg_o", "").strip()
                    desc_val  = str(description_o or "").strip()
                    address_val = str(st.session_state.get("address_o", "")).strip()
                    created_by_o = DEFAULT_CREATED_BY
                    channel_o_val = st.session_state.get("channel_o") or ""

                    if not osp_val:  missing.append("OSP Appointment Type")
                    if not ont_val:  missing.append("ONT ID")
                    if not city_sel: missing.append("City")
                    if not sn_val:   missing.append("Second Number")
                    if not issue_sel:missing.append("Issue Type")
                    if not call_sel: missing.append("Call Type")
                    if not fttg_sel: missing.append("FTTG")
                    if not desc_val: missing.append("Description")
                    if not address_val: missing.append("Address")
                    if not channel_o_val: missing.append("Channel")

                    if missing:
                        _color_missing_labels(missing)
                        st.error("Please fill in all required fields: " + ", ".join(missing))
                    else:
                        add_ticket({
                            "ticket_group": "OSP Appointments",
                            "osp_type": osp_val,
                            "ont_id": ont_val,
                            "city": city_sel,
                            "second_number": sn_val,
                            "issue_type": issue_sel,
                            "call_type": call_sel,
                            "channel": channel_o_val,
                            "fttg": fttg_sel,
                            "description": desc_val,
                            "address": address_val,
                            "created_by": created_by_o,
                        })
                        st.success("OSP appointment ticket added.")
                        st.session_state['_ticket_saved_message'] = "OSP appointment ticket created successfully."
                        st.session_state['_show_ticket_saved_dialog'] = True
                        st.session_state.show_new_form = False
                        st.rerun()
            # Developer notes: render OSP appointments markdown below the form inside an expander
            try:
                _dev_md_o = load_implementation_note("osp_appointments.md")
            except Exception:
                _dev_md_o = None
            if _dev_md_o:
                with st.expander("Implementation notes (OSP Appointments)", expanded=False):
                    st.markdown('<div class="implementation-notes-tag"></div>', unsafe_allow_html=True)
                    st.markdown(_dev_md_o)
# ---------- Tickets table (only render in Call Tickets > Tickets) ----------
if (
    st.session_state.active_tab == "Call Tickets"
    and st.session_state.active_subtab == "Tickets"
    and not _is_popup_view()
):
    detail_view_id = st.session_state.get("_ticket_detail_id")
    if st.session_state.get("_show_ticket_detail") and detail_view_id is not None:
        st.session_state["show_new_form"] = False
        _render_ticket_detail_page(detail_view_id)
        st.stop()

    df = st.session_state.tickets_df.copy()

    st.markdown("### Tickets")
    search = st.text_input("Search", placeholder="Search ONT / Description…", label_visibility="collapsed")

    if search:
        s = search.lower()
        def row_match(row):
            fields = [
                "ont_id", "description", "complaint_type", "activity_inquiry_type",
                "kurdtel_service_status", "online_game",
                "osp_type", "second_number", "olt", "city", "issue_type", "fttg", "created_by", "address"
            ]
            return any(str(row.get(f, "")).lower().find(s) >= 0 for f in fields)
        df = df[df.apply(row_match, axis=1)]

    # Prepare Ticket Group options for filtering (used under tabs)
    # Always include preset groups so the dropdown has values even when no records exist
    PRESET_TICKET_GROUPS = [
        "Activities & Inquiries",
        "Complaints",
        "OSP Appointments",
    ]
    try:
        _groups_series = df.get("ticket_group") if isinstance(df, pd.DataFrame) else None
        _from_data = (
            _groups_series.dropna().astype(str).tolist() if _groups_series is not None else []
        )
    except Exception:
        _from_data = []
    _combined = PRESET_TICKET_GROUPS + sorted(set(g for g in _from_data if g))
    _seen = set()
    _groups = []
    for g in _combined:
        gs = str(g).strip()
        if gs and gs not in _seen:
            _groups.append(gs)
            _seen.add(gs)
    TICKET_GROUP_FILTER_OPTIONS = ["All"] + _groups
    st.session_state["_ticket_group_choices_fallback"] = _groups

    # Helpers: export payload and UI (Excel icon with tooltip)
    def _export_payload(df_in: pd.DataFrame) -> tuple[bytes, str, str]:
        import io
        b = io.BytesIO()
        # Try Excel first; fall back to CSV if engines aren't available
        try:
            with pd.ExcelWriter(b) as writer:  # engine auto-detect
                df_in.to_excel(writer, index=False, sheet_name="Tickets")
            b.seek(0)
            return b.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
        except Exception:
            try:
                return df_in.to_csv(index=False).encode("utf-8-sig"), "text/csv", "csv"
            except Exception:
                return b"", "application/octet-stream", "bin"

    def _ticket_group_filter_ui(
        loc_key: str,
        df_export: pd.DataFrame | None = None,
        filename_prefix: str = "tickets",
        show_filter: bool = True,
        status_options: list[str] | None = None,
        status_state_key: str | None = None,
        status_label: str = "Complaint Status",
    ):
        filter_container = st.container()
        with filter_container:
            if show_filter:
                _current = st.session_state.get(f"ticket_group_filter_{loc_key}")
                if _current is None:
                    _current = st.session_state.get("ticket_group_filter", "All")
                if _current not in TICKET_GROUP_FILTER_OPTIONS:
                    _current = "All"
                sel = st.selectbox(
                    "Ticket Group",
                    options=TICKET_GROUP_FILTER_OPTIONS,
                    index=TICKET_GROUP_FILTER_OPTIONS.index(_current),
                    key=f"ticket_group_filter_{loc_key}",
                )
                st.session_state["ticket_group_filter"] = sel
            elif status_options and status_state_key:
                _opts = ["All"] + [o for o in status_options if o]
                _cur = st.session_state.get(status_state_key, "All")
                if _cur not in _opts:
                    _cur = "All"
                _widget_key = f"{status_state_key}_{loc_key}"
                sel2 = st.selectbox(
                    status_label,
                    options=_opts,
                    index=_opts.index(_cur),
                    key=_widget_key,
                )
                st.session_state[status_state_key] = sel2
            else:
                st.write("")

        actions_container = st.container()
        with actions_container:
            export_col, update_col, _ = st.columns([1, 1, 6], gap="small")

            with export_col:
                button_key = f"export_btn_{loc_key}"
                if df_export is not None and not df_export.empty:
                    data, mime, ext = _export_payload(df_export)
                    ts = _now_local().strftime("%Y%m%d_%H%M%S")
                    fname = f"{filename_prefix}_{ts}.{ext}"
                    st.download_button(
                        "Export",
                        data=data,
                        file_name=fname,
                        mime=mime,
                        key=button_key,
                        use_container_width=False,
                    )
                else:
                    st.download_button(
                        "Export",
                        data=b"",
                        file_name="tickets.csv",
                        key=button_key,
                        disabled=True,
                        use_container_width=False,
                    )

            with update_col:
                update_container = st.container()

        return update_container

    def _apply_ticket_group_filter(df_in: pd.DataFrame, state_key: str = "ticket_group_filter") -> pd.DataFrame:
        df_filtered = df_in
        try:
            sel = st.session_state.get(state_key)
            if sel is None:
                sel = st.session_state.get("ticket_group_filter", "All")
            else:
                st.session_state["ticket_group_filter"] = sel
            if sel is None:
                sel = "All"
            if sel and sel != "All" and "ticket_group" in df_in.columns:
                df_filtered = df_in[df_in["ticket_group"].astype(str) == sel]
        except Exception:
            df_filtered = df_in

        session_key = f"{state_key}_choices"
        try:
            choices = sorted(
                df_filtered.get("ticket_group", pd.Series(dtype=str)).dropna().astype(str).unique(),
                key=str.lower,
            )
            st.session_state[session_key] = choices
            st.session_state["_ticket_group_choices"] = choices
        except Exception:
            st.session_state.pop(session_key, None)
            st.session_state["_ticket_group_choices"] = []
        return df_filtered

    def _normalize_ticket_id(raw) -> int | None:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                if item is not None:
                    raw = item
                    break
            else:
                return None
        if isinstance(raw, dict):
            raw = raw.get("id") or raw.get("value")
        try:
            if isinstance(raw, bool):
                return None
            if isinstance(raw, int):
                return int(raw)
            if isinstance(raw, float):
                if pd.isna(raw):
                    return None
                return int(raw)
        except Exception:
            pass
        try:
            s = str(raw).strip()
        except Exception:
            return None
        if not s:
            return None
        s = s.strip("[]'\"")
        if not s or s.lower() in {"none", "nan", ""}:
            return None
        try:
            return int(s)
        except Exception:
            try:
                return int(float(s))
            except Exception:
                return None

    def _extract_selected_ticket_ids(grid_resp) -> set[int]:
        if grid_resp is None:
            return set()

        rows_candidate = None
        if isinstance(grid_resp, dict):
            rows_candidate = grid_resp.get("selected_rows")
            if rows_candidate is None:
                rows_candidate = grid_resp.get("selectedRows")
        else:
            rows_candidate = getattr(grid_resp, "selected_rows", None)

        if rows_candidate is None:
            return set()

        if isinstance(rows_candidate, pd.DataFrame):
            rows_iterable = rows_candidate.to_dict(orient="records")
        elif isinstance(rows_candidate, list):
            rows_iterable = rows_candidate
        else:
            try:
                rows_iterable = list(rows_candidate)
            except Exception:
                rows_iterable = []

        ticket_ids: set[int] = set()
        for row in rows_iterable:
            if isinstance(row, dict):
                tid = _normalize_ticket_id(row.get("id"))
            else:
                tid = _normalize_ticket_id(row)
            if tid is not None:
                ticket_ids.add(tid)
        return ticket_ids

    def _render_bulk_update_controls(container, selected_ids: set[int], origin_key: str):
        if container is None:
            return
        placeholder = container.empty()
        count = len(selected_ids)
        label = f"Update ({count})" if count else "Update"
        disabled = count == 0
        if placeholder.button(
            label,
            key=f"bulk_update_btn_{origin_key}",
            use_container_width=False,
            disabled=disabled,
        ):
            st.session_state["_bulk_update_ticket_ids"] = sorted(selected_ids)
            st.session_state["_bulk_update_origin"] = origin_key
            st.session_state["_bulk_update_dialog_open"] = True

    def _get_ticket_group_choices() -> list[str]:
        dynamic = st.session_state.get("_ticket_group_choices") or []
        fallback = st.session_state.get("_ticket_group_choices_fallback", [])

        seen: set[str] = set()
        merged: list[str] = []

        def _add_options(options: list[str] | tuple[str, ...]) -> None:
            for opt in options:
                if not opt:
                    continue
                opt_str = str(opt).strip()
                if not opt_str:
                    continue
                key = opt_str.lower()
                if key == "all":
                    continue
                if key not in seen:
                    merged.append(opt_str)
                    seen.add(key)

        if isinstance(dynamic, (list, tuple)):
            _add_options(list(dynamic))
        if isinstance(fallback, (list, tuple)):
            _add_options(list(fallback))
        _add_options([g for g in TICKET_GROUP_FILTER_OPTIONS if g])

        return merged

    BULK_UPDATE_GROUP_PROMPT = "— Select Ticket Group —"


    BULK_UPDATE_DEPENDENCIES = {
        "Activities & Inquiries": [
            {
                "label": "Activity / Inquiry Type",
                "options": lambda: ACTIVITY_TYPES,
                "field": "activity_inquiry_type",
                "state_key": "bulk_update_activity_choice",
                "required": True,
            },
        ],
        "Complaints": [
            {
                "label": "Complaint Type",
                "options": lambda: COMPLAINT_TYPES,
                "field": "complaint_type",
                "state_key": "bulk_update_complaint_choice",
                "required": True,
            },
            {
                "label": "Employee Suggestion",
                "options": lambda: EMP_SUGGESTION,
                "field": "employee_suggestion",
                "state_key": "bulk_update_employee_suggestion_choice",
                "required": False,
            },
        ],
        "OSP Appointments": [
            {
                "label": "OSP Appointment Type",
                "options": lambda: OSP_TYPES,
                "field": "osp_type",
                "state_key": "bulk_update_osp_choice",
                "required": True,
            },
        ],
    }

    def _resolve_dependency_configs(group: str) -> list[dict[str, Any]]:
        entries = BULK_UPDATE_DEPENDENCIES.get(group)
        if not entries:
            return []

        resolved: list[dict[str, Any]] = []
        for meta in entries:
            if not isinstance(meta, dict):
                continue
            options_source = meta.get("options", [])
            try:
                options = options_source() if callable(options_source) else options_source
            except Exception:
                options = []
            sanitized = []
            for opt in options or []:
                opt_str = str(opt).strip()
                if opt_str:
                    sanitized.append(opt_str)

            label = str(meta.get("label") or "Selection").strip()
            required = bool(meta.get("required", True))
            state_key = meta.get("state_key") or f"{label.lower().replace(' ', '_')}_choice"

            resolved.append(
                {
                    "label": label,
                    "display_label": f"{label} (optional)" if not required else label,
                    "options": sanitized,
                    "field": meta.get("field"),
                    "state_key": state_key,
                    "required": required,
                }
            )

        return resolved

    def _reset_bulk_update_dependency_state() -> None:
        for configs in BULK_UPDATE_DEPENDENCIES.values():
            if not isinstance(configs, (list, tuple)):
                continue
            for meta in configs:
                if not isinstance(meta, dict):
                    continue
                state_key = meta.get("state_key")
                if not state_key:
                    continue
                st.session_state.pop(state_key, None)
                st.session_state.pop(f"{state_key}_widget", None)

    def _toast_warning(message: str, container=None) -> None:
        target = container if container is not None else st
        try:
            target.warning(message)
        except Exception:
            st.warning(message)

    def _render_bulk_dependency_select(
        label: str,
        options: list[str],
        state_key: str,
        *,
        required: bool = True,
    ) -> str | None:
        if not options:
            st.warning(f"No options found for {label}.")
            st.session_state.pop(state_key, None)
            st.session_state.pop(f"{state_key}_widget", None)
            return None

        if required:
            prompt = f"— Select {label} —"
        else:
            prompt = f"— Keep existing {label} —"
        widget_key = f"{state_key}_widget"
        current_value = st.session_state.get(state_key)
        default_value = current_value if current_value in options else prompt
        st.session_state.setdefault(widget_key, default_value)
        if st.session_state[widget_key] not in ([prompt] + options):
            st.session_state[widget_key] = default_value

        selected_raw = st.selectbox(label, [prompt] + options, key=widget_key)
        value = None if selected_raw == prompt else selected_raw
        st.session_state[state_key] = value
        return value

    def _clear_bulk_update_state():
        for _key in [
            "_bulk_update_ticket_ids",
            "_bulk_update_origin",
            "_bulk_update_dialog_open",
            "_bulk_update_choice",
            "bulk_update_ticket_group_choice",
            "bulk_update_ticket_group_choice_widget",
            "_bulk_update_success",
        ]:
            if _key in st.session_state:
                st.session_state.pop(_key, None)
        _reset_bulk_update_dependency_state()

    def _bulk_update_ticket_group(
        ticket_ids: list[int],
        new_group: str,
        extra_updates: dict[str, Any] | None = None,
    ) -> int:
        if not ticket_ids or not new_group:
            return 0
        normalized = sorted({
            tid for tid in (_normalize_ticket_id(t) for t in ticket_ids) if tid is not None
        })
        if not normalized:
            return 0
        engine = get_tickets_engine()
        table = _reflect_tickets_table()
        values_payload: dict[str, Any] = {"ticket_group": new_group}
        if extra_updates:
            for key, val in extra_updates.items():
                values_payload[key] = val
        try:
            with engine.begin() as conn:
                result = conn.execute(
                    update(table)
                    .where(table.c.id.in_(normalized))
                    .values(**values_payload)
                )
            _refresh_tickets_df(engine)
        except (SQLAlchemyError, Exception) as exc:
            logging.exception("Bulk ticket group update failed: %s", exc)
            raise
        try:
            updated = result.rowcount if result is not None else None
        except Exception:
            updated = None
        return updated if updated not in (None, 0) else len(normalized)

    @st.dialog("Update Ticket Group")
    def _bulk_update_ticket_group_dialog():
        selected_ids = st.session_state.get("_bulk_update_ticket_ids") or []
        ticket_count = len(selected_ids)
        choices = _get_ticket_group_choices()

        success_info = st.session_state.get("_bulk_update_success")
        if success_info:
            updated_count = success_info.get("updated", ticket_count)
            message = f"Updated {updated_count} ticket{'s' if updated_count != 1 else ''}"
            group_label = success_info.get("group")
            detail = success_info.get("detail")
            if group_label:
                message += f" to {group_label}"
            if detail:
                message += f" ({detail})"
            message += "."
            st.success(message)
            if st.button("Close", key="bulk_update_after_success_close", use_container_width=True):
                _clear_bulk_update_state()
                st.rerun()
            return

        if not selected_ids:
            st.info("Select at least one ticket to update.")
            if st.button("Close", key="bulk_update_close", use_container_width=True):
                _clear_bulk_update_state()
                st.rerun()
            return

        st.write(f"Updating {ticket_count} ticket{'s' if ticket_count != 1 else ''}.")

        if not choices:
            st.warning("No ticket groups available. Adjust filters and try again.")
            if st.button("Got it", key="bulk_update_no_choices", use_container_width=True):
                _clear_bulk_update_state()
                st.rerun()
            return

        options_with_prompt = [BULK_UPDATE_GROUP_PROMPT] + choices
        widget_key = "bulk_update_ticket_group_choice_widget"
        prev_group = st.session_state.get("bulk_update_ticket_group_choice")
        default_value = prev_group if prev_group in choices else BULK_UPDATE_GROUP_PROMPT
        st.session_state.setdefault(widget_key, default_value)
        if st.session_state[widget_key] not in options_with_prompt:
            st.session_state[widget_key] = default_value

        selected_group_raw = st.selectbox(
            "New Ticket Group",
            options_with_prompt,
            key=widget_key,
        )

        new_group = None if selected_group_raw == BULK_UPDATE_GROUP_PROMPT else selected_group_raw
        if new_group != prev_group:
            _reset_bulk_update_dependency_state()
        st.session_state["bulk_update_ticket_group_choice"] = new_group

        dependency_configs: list[dict[str, Any]] = []
        if new_group:
            dependency_configs = _resolve_dependency_configs(new_group)
            for config in dependency_configs:
                options = config.get("options", [])
                label = config.get("display_label") or config.get("label", "Selection")
                state_key = config.get("state_key")
                required = bool(config.get("required", True))

                if options:
                    _render_bulk_dependency_select(
                        label,
                        options,
                        state_key,
                        required=required,
                    )
                else:
                    if required:
                        st.warning(
                            f"No options available for {config.get('label')}. Update the lookup data before continuing."
                        )
                    else:
                        st.info(
                            f"No options available for {config.get('label')}. Existing values will be left unchanged."
                        )

        warning_placeholder = st.empty()

        col_cancel, col_confirm = st.columns(2)
        with col_cancel:
            if st.button("Cancel", key="bulk_update_cancel", use_container_width=True):
                _clear_bulk_update_state()
                st.rerun()
        with col_confirm:
            if st.button("Apply", key="bulk_update_apply", use_container_width=True):
                if not new_group:
                    _toast_warning("Select a ticket group before applying.", container=warning_placeholder)
                    return

                extra_updates: dict[str, Any] | None = None
                detail_parts: list[str] = []
                if dependency_configs:
                    for config in dependency_configs:
                        state_key = config.get("state_key")
                        label = config.get("label", "Selection")
                        value = st.session_state.get(state_key)
                        required = bool(config.get("required", True))
                        options = config.get("options", [])
                        field_name = config.get("field")

                        if required and not value:
                            if options:
                                _toast_warning(
                                    f"Select {label} before applying.",
                                    container=warning_placeholder,
                                )
                            else:
                                _toast_warning(
                                    f"No {label} options are available. Update the lookup data before retrying.",
                                    container=warning_placeholder,
                                )
                            return

                        if value and field_name:
                            extra_updates = extra_updates or {}
                            extra_updates[field_name] = value
                            detail_parts.append(f"{label}: {value}")

                try:
                    updated = _bulk_update_ticket_group(selected_ids, new_group, extra_updates)
                except Exception:
                    st.error("Couldn't update the selected tickets. Please try again.")
                    return
                success_payload = {"updated": updated, "group": new_group}
                if detail_parts:
                    success_payload["detail"] = "; ".join(detail_parts)
                st.session_state["_bulk_update_success"] = success_payload
                st.rerun()


def _render_tickets_grid(
    df_input,
    ag_key: str | None = None,
    *,
    width_reference_df: pd.DataFrame | None = None,
):
    display_cols = [
        c for c in GRID_DEFAULT_DISPLAY_COLUMNS
        if c == "ticket_type" or c in df_input.columns
    ]
    if "comment_count" in df_input.columns and "comment_count" not in display_cols:
        display_cols.append("comment_count")
    if not display_cols:
        display_cols = [c for c in df_input.columns if c != "_edit_request"]

    ticket_type_source_cols = ["activity_inquiry_type", "complaint_type", "osp_type"]

    source_cols: list[str] = []
    for col in display_cols:
        if col != "ticket_type" and col in df_input.columns and col not in source_cols:
            source_cols.append(col)
    for col in ticket_type_source_cols:
        if col in df_input.columns and col not in source_cols:
            source_cols.append(col)
    if "comment_count" in df_input.columns and "comment_count" not in source_cols:
        source_cols.append("comment_count")
    if not source_cols:
        source_cols = [c for c in ticket_type_source_cols if c in df_input.columns]

    use_cols = source_cols if source_cols else list(df_input.columns)

    # Some data sources (or an empty store) may not have an 'id' column yet.
    # Try to sort by 'id' when present; otherwise fall back to a safe deterministic order
    try:
        if "id" in use_cols:
            df_subset = df_input[use_cols].sort_values("id", ascending=False).reset_index(drop=True)
        else:
            # ensure we still include an 'id' column for downstream logic
            df_subset = df_input[use_cols].reset_index(drop=True)
            if "id" not in df_subset.columns:
                df_subset.insert(0, "id", pd.Series(dtype=object))
    except KeyError:
        # unexpected missing column; fallback to safe empty frame with expected columns
        df_subset = df_input.reindex(columns=use_cols).copy()
        if "id" not in df_subset.columns:
            df_subset.insert(0, "id", pd.Series(dtype=object))
        df_subset = df_subset.reset_index(drop=True)

    if "ticket_group" in df_subset.columns or any(col in df_subset.columns for col in ticket_type_source_cols):
        ticket_type_series = pd.Series("", index=df_subset.index, dtype=object)
        if "ticket_group" in df_subset.columns:
            group_values = df_subset["ticket_group"].fillna("")
        else:
            group_values = pd.Series([""] * len(df_subset), index=df_subset.index, dtype=object)

        if "activity_inquiry_type" in df_subset.columns:
            mask_ai = group_values == "Activities & Inquiries"
            ticket_type_series.loc[mask_ai] = df_subset.loc[mask_ai, "activity_inquiry_type"].fillna("")
        if "complaint_type" in df_subset.columns:
            mask_comp = group_values == "Complaints"
            ticket_type_series.loc[mask_comp] = df_subset.loc[mask_comp, "complaint_type"].fillna("")
        if "osp_type" in df_subset.columns:
            mask_osp = group_values == "OSP Appointments"
            ticket_type_series.loc[mask_osp] = df_subset.loc[mask_osp, "osp_type"].fillna("")

        for src in ticket_type_source_cols:
            if src in df_subset.columns:
                empties = ticket_type_series.astype(str).str.strip().eq("") | ticket_type_series.isna()
                ticket_type_series.loc[empties] = df_subset.loc[empties, src].fillna("")

        ticket_type_series = ticket_type_series.where(ticket_type_series.notna(), "")
        ticket_type_series = ticket_type_series.astype(str).str.strip()
        ticket_type_series = ticket_type_series.replace({"nan": "", "None": ""})
        df_subset["ticket_type"] = ticket_type_series
    else:
        df_subset["ticket_type"] = ""

    final_display_cols = [c for c in display_cols if c in df_subset.columns]
    if "ticket_type" in display_cols and "ticket_type" not in final_display_cols:
        final_display_cols.append("ticket_type")
    df_sorted = df_subset[final_display_cols].copy() if final_display_cols else df_subset.copy()
    display_cols = final_display_cols

    # View df + hidden column for inline edit trigger
    df_view = df_sorted.copy()
    if "id" in df_view.columns:
        df_view["id"] = df_view["id"].astype(str)
    df_view.insert(0, "Select", "")
    df_view.insert(1, "Edit", "")
    if "_edit_request" not in df_view.columns:
        df_view["_edit_request"] = ""
    if "_detail_request" not in df_view.columns:
        df_view["_detail_request"] = ""

    gb = GridOptionsBuilder.from_dataframe(df_view, enableRowGroup=False, enableValue=False, enablePivot=False)
    # Disable column filters globally (no filter UI on any column)
    gb.configure_default_column(
        editable=False,
        resizable=True,
        filter=False,
        suppressMenu=True,
        menuTabs=[],
    )
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
        rowMultiSelectWithClick=True,
        suppressRowDeselection=False,
    )

    gb.configure_column(
        "Select",
        headerName="",
        pinned="left",
        checkboxSelection=True,
        headerCheckboxSelection=True,
        headerCheckboxSelectionFilteredOnly=True,
        sortable=False,
    )

    ellipsis_cell_style = {
        "whiteSpace": "nowrap",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    }

    if "description" in df_view.columns:
        gb.configure_column(
            "description",
            cellStyle=ellipsis_cell_style,
            tooltipField="description",
        )
    if "address" in df_view.columns:
        gb.configure_column(
            "address",
            cellStyle=ellipsis_cell_style,
            tooltipField="address",
        )

    _header_labels = {
        "id": "#",
        "created_at": "Created At",
        "ticket_group": "Ticket Group",
        "ont_id": "ONT ID",
        "call_type": "Call Type",
        "ticket_type": "Ticket Type",
        "employee_suggestion": "Employee Suggestion",
        "device_location": "Device Location",
        "root_cause": "Root Cause",
        "ont_model": "ONT Model",
        "online_game": "Online Game",
        "complaint_status": "Complaint Status",
        "refund_type": "Refund Type",
        "kurdtel_service_status": "Kurdtel Service Status",
        "city": "City",
        "issue_type": "Issue Type",
        "fttg": "FTTG",
        "olt": "OLT",
        "second_number": "Second Number",
        "created_by": "Created By",
        "address": "Address",
        "description": "Description",
        "fttx_job_status": "FTTX Job Status",
        "fttx_job_remarks": "FTTX Job Remarks",
        "fttx_cancel_reason": "FTTX Cancel Reason",
        "callback_status": "Call-Back Status",
        "callback_reason": "Call-Back Reason",
        "followup_status": "Follow-Up Status",
    "visit_required": "Visit Required",
        "reminder_enabled": "Reminder Set",
        "reminder_recipient": "Reminder Recipient",
        "reminder_note": "Reminder Note",
        "reminder_at": "Reminder At",
        "channel": "Channel",
    }

    for _col, _label in _header_labels.items():
        if _col in df_view.columns:
            gb.configure_column(_col, headerName=_label)

    if "ticket_group" in df_view.columns:
        ticket_group_renderer = _make_badge_renderer_js(
            "TicketGroupBadgeRenderer",
            TICKET_GROUP_BADGE_COLORS,
            default_color="#2563eb",
        )
        gb.configure_column(
            "ticket_group",
            headerName=_header_labels.get("ticket_group", "Ticket Group"),
            cellRenderer=ticket_group_renderer,
        )

    if "complaint_status" in df_view.columns:
        complaint_status_renderer = _make_badge_renderer_js(
            "ComplaintStatusBadgeRenderer",
            COMPLAINT_STATUS_BADGE_COLORS,
            default_color="#6b7280",
        )
        gb.configure_column(
            "complaint_status",
            headerName=_header_labels.get("complaint_status", "Complaint Status"),
            cellRenderer=complaint_status_renderer,
        )

    if "reminder_at" in df_view.columns:
        gb.configure_column(
            "reminder_at",
            headerName=_header_labels.get("reminder_at", "Reminder At"),
        )

    gb.configure_column("_edit_request", hide=True)
    gb.configure_column("_detail_request", hide=True)
    if "comment_count" in df_view.columns:
        gb.configure_column("comment_count", hide=True)

    gb.configure_column(
        "Edit",
        headerName="",
        pinned="left",
        sortable=False,
        cellRenderer=JsCode("""
            class IconCellRenderer {
                init(params){
                    const idRaw = params && params.data ? params.data.id : null;
                    const id = (idRaw === null || idRaw === undefined) ? '' : String(idRaw);
                    const commentCountRaw = params && params.data ? params.data.comment_count : null;
                    const commentCount = (commentCountRaw === null || commentCountRaw === undefined) ? 0 : Number(commentCountRaw);
                    const hasComments = !Number.isNaN(commentCount) && commentCount > 0;

                    if (params && params.eGridCell) {
                        try {
                            params.eGridCell.innerHTML = '';
                            params.eGridCell.style.display = 'flex';
                            params.eGridCell.style.alignItems = 'center';
                            params.eGridCell.style.justifyContent = 'center';
                        } catch (e) {}
                    }

                    const wrapper = document.createElement('span');
                    wrapper.className = 'grid-action-icons';
                    wrapper.style.display = 'flex';
                    wrapper.style.alignItems = 'center';
                    wrapper.style.justifyContent = 'center';
                    wrapper.style.gap = '0.5rem';
                    wrapper.style.width = '100%';
                    wrapper.style.height = '100%';

                    const editIcon = document.createElement('span');
                    editIcon.className = 'grid-action-icon grid-action-icon--edit';
                    editIcon.textContent = '✏️';
                    editIcon.title = 'Edit ticket';
                    editIcon.setAttribute('role', 'button');
                    editIcon.setAttribute('tabindex', '0');

                    const detailIcon = document.createElement('span');
                    detailIcon.className = 'grid-action-icon grid-action-icon--detail';
                    detailIcon.textContent = '📄';
                    detailIcon.title = 'View ticket details';
                    detailIcon.setAttribute('role', 'button');
                    detailIcon.setAttribute('tabindex', '0');

                    const commentIcon = hasComments ? document.createElement('span') : null;
                    if (commentIcon) {
                        commentIcon.className = 'grid-action-icon grid-action-icon--comment';
                        commentIcon.textContent = '💬';
                        commentIcon.title = 'View comments';
                        commentIcon.setAttribute('role', 'button');
                        commentIcon.setAttribute('tabindex', '0');
                    }

                    const dispatchEdit = (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        if (!id || !params || !params.node) {
                            return;
                        }
                        try {
                            params.node.setDataValue('_edit_request', id + '|' + String(Date.now()));
                            params.api.dispatchEvent({ type: 'modelUpdated' });
                        } catch (e) {}
                    };

                    const dispatchDetail = (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        if (!id || !params || !params.node) {
                            return;
                        }
                        try {
                            params.node.setDataValue('_detail_request', id + '|' + String(Date.now()));
                            params.api.dispatchEvent({ type: 'modelUpdated' });
                        } catch (e) {}
                    };

                    editIcon.addEventListener('click', dispatchEdit);
                    detailIcon.addEventListener('click', dispatchDetail);
                    if (commentIcon) {
                        commentIcon.addEventListener('click', dispatchDetail);
                    }
                    editIcon.addEventListener('keydown', (event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                            dispatchEdit(event);
                        }
                    });
                    detailIcon.addEventListener('keydown', (event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                            dispatchDetail(event);
                        }
                    });
                    if (commentIcon) {
                        commentIcon.addEventListener('keydown', (event) => {
                            if (event.key === 'Enter' || event.key === ' ') {
                                dispatchDetail(event);
                            }
                        });
                    }

                    wrapper.appendChild(editIcon);
                    if (commentIcon) {
                        wrapper.appendChild(commentIcon);
                    }
                    wrapper.appendChild(detailIcon);
                    this.eGui = wrapper;
                }
                getGui(){ return this.eGui; }
            }
        """),
    )

    side_bar_config = {
        "toolPanels": [
            {
                "id": "columns",
                "labelDefault": "Columns",
                "labelKey": "columns",
                "iconKey": "columns",
                "toolPanel": "agColumnsToolPanel",
                "minWidth": 225,
                "toolPanelParams": {
                    "suppressRowGroups": True,
                    "suppressValues": True,
                    "suppressPivots": True,
                    "suppressPivotMode": True,
                    "suppressColumnFilter": True,
                },
            }
        ],
        "position": "right",
        "hiddenByDefault": False,
    }

    gb.configure_grid_options(
        onFirstDataRendered=JsCode(
            """
            function(params){
                if (!params || !params.columnApi || !params.columnApi.autoSizeAllColumns) {
                    return;
                }

                const autoSize = () => {
                    try { params.columnApi.autoSizeAllColumns(); } catch (e) {}
                    try { params.api.resetRowHeights(); } catch (e) {}
                    try { params.api.closeToolPanel(); } catch (e) {}
                };

                window.requestAnimationFrame(autoSize);
                [0, 120, 400].forEach((delay) => {
                    window.setTimeout(autoSize, delay);
                });
            }
            """
        ),
        onGridSizeChanged=JsCode(
            """
            function(params){
                if (!params || !params.columnApi || !params.columnApi.autoSizeAllColumns) {
                    return;
                }

                const run = () => {
                    try { params.columnApi.autoSizeAllColumns(); } catch (e) {}
                    try { params.api.resetRowHeights(); } catch (e) {}
                };

                window.requestAnimationFrame(run);
                window.setTimeout(run, 150);
            }
            """
        ),
        suppressRowClickSelection=True,
        rowSelection="multiple",
        rememberSelection=True,
        sideBar=side_bar_config,
        suppressHeaderMenuButton=True,
        suppressColumnMenu=True,
        columnMenu="none",
    )

    grid_options = gb.build()
    grid_options.setdefault("columnMenu", "none")

    default_col_def = grid_options.setdefault("defaultColDef", {})
    default_col_def.setdefault("suppressMenu", True)
    default_col_def.setdefault("menuTabs", [])
    default_col_def.setdefault("suppressHeaderMenuButton", True)

    col_defs = grid_options.get("columnDefs", [])
    if isinstance(col_defs, list):
        for _col_def in col_defs:
            if isinstance(_col_def, dict):
                _col_def.setdefault("suppressMenu", True)
                _col_def.setdefault("menuTabs", [])
                _col_def.setdefault("suppressHeaderMenuButton", True)

    auto_group_def = grid_options.get("autoGroupColumnDef")
    if isinstance(auto_group_def, dict):
        auto_group_def.setdefault("suppressMenu", True)
        auto_group_def.setdefault("menuTabs", [])
        auto_group_def.setdefault("suppressHeaderMenuButton", True)

    update_mode = GridUpdateMode.MODEL_CHANGED
    if hasattr(GridUpdateMode, "SELECTION_CHANGED"):
        try:
            update_mode = GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED
        except TypeError:
            update_mode = GridUpdateMode.SELECTION_CHANGED

    grid_resp = AgGrid(
        df_view,
        gridOptions=grid_options,
        key=ag_key,
        height=520,
        fit_columns_on_grid_load=False,
        update_mode=update_mode,
        data_return_mode=DataReturnMode.AS_INPUT,
        allow_unsafe_jscode=True,
        theme="balham",
        enable_enterprise_modules=True,
    )

    # Detect inline edit/detail icon clicks and respond accordingly
    try:
        _data = grid_resp.get('data') if grid_resp else None
        if _data is not None:
            import pandas as _pd
            if isinstance(_data, _pd.DataFrame):
                rows = _data.to_dict(orient="records")
            elif isinstance(_data, list):
                rows = _data
            else:
                rows = []
            def _token_ts(flag):
                try:
                    s = str(flag)
                    return float(s.split('|', 1)[1]) if '|' in s else 0.0
                except Exception:
                    return 0.0
            def _coerce_int(v):
                try:
                    return int(v)
                except Exception:
                    try:
                        if isinstance(v, list) and v:
                            return int(str(v[0]).strip().strip("[]'\" "))
                    except Exception:
                        pass
                    try:
                        s = str(v).strip()
                        if s.startswith('[') and s.endswith(']'):
                            s = s.strip('[]').strip().strip("'\"")
                        return int(float(s))
                    except Exception:
                        return None

            edit_candidates = [r for r in rows if str(r.get('_edit_request', '')).strip() not in ('', '0', 'false', 'None')]
            edit_req = max(edit_candidates, key=lambda r: _token_ts(r.get('_edit_request')), default=None)
            if edit_req:
                _flag = str(edit_req.get('_edit_request', '')).strip()
                _rid = edit_req.get('id')
                _tid = _coerce_int(_rid)
                if _tid is not None and st.session_state.get('_last_edit_token') != _flag:
                    st.session_state['_last_edit_token'] = _flag
                    st.session_state['_edit_open_id'] = _tid
                    edit_ticket_dialog(_tid)

            detail_candidates = [r for r in rows if str(r.get('_detail_request', '')).strip() not in ('', '0', 'false', 'None')]
            detail_req = max(detail_candidates, key=lambda r: _token_ts(r.get('_detail_request')), default=None)
            if detail_req:
                _detail_flag = str(detail_req.get('_detail_request', '')).strip()
                _detail_rid = detail_req.get('id')
                _detail_tid = _coerce_int(_detail_rid)
                if _detail_tid is not None and st.session_state.get('_last_detail_token') != _detail_flag:
                    st.session_state['_last_detail_token'] = _detail_flag
                    normalized = _activate_ticket_detail(_detail_tid)
                    if normalized is not None:
                        try:
                            st.query_params['ticket_detail'] = str(normalized)
                        except Exception:
                            pass
                    st.rerun()
    except Exception as _e:
        import logging as _lg
        _lg.exception("Inline edit open failed: %s", _e)

    return grid_resp


if (
    st.session_state.active_tab == "Call Tickets"
    and st.session_state.active_subtab == "Tickets"
    and not _is_popup_view()
):
    # Render tickets view: use a controlled selector so only the active grid is created.
    creator_name = CURRENT_USER_NAME
    view_choices = ["My Tickets", "All Tickets", "Kurdtel"]
    default_choice = st.session_state.get("tickets_view_choice", "My Tickets")
    if default_choice not in view_choices:
        default_choice = "My Tickets"
    selection = st.radio(
        "Select ticket view",
        options=view_choices,
        index=view_choices.index(default_choice),
        horizontal=True,
        key="tickets_view_choice",
        label_visibility="collapsed",
    )

    if selection == "My Tickets":
        if "created_by" in df.columns:
            df_my = df[df["created_by"].astype(str).str.lower() == creator_name.lower()]
        elif "assigned_to" in df.columns:
            df_my = df[df["assigned_to"].astype(str).str.lower() == creator_name.lower()]
        else:
            df_my = df.iloc[0:0]
        df_my = _apply_ticket_group_filter(df_my, state_key="ticket_group_filter_my")
        action_col_my = _ticket_group_filter_ui("my", df_export=df_my, filename_prefix="my_tickets")
        grid_resp_my = _render_tickets_grid(
            df_my,
            ag_key="aggrid_my",
            width_reference_df=df,
        )
        selected_my = _extract_selected_ticket_ids(grid_resp_my)
        _render_bulk_update_controls(action_col_my, selected_my, "my")

    elif selection == "All Tickets":
        df_all = _apply_ticket_group_filter(df, state_key="ticket_group_filter_all")
        action_col_all = _ticket_group_filter_ui("all", df_export=df_all, filename_prefix="all_tickets")
        grid_resp_all = _render_tickets_grid(
            df_all,
            ag_key="aggrid_all",
            width_reference_df=df,
        )
        selected_all = _extract_selected_ticket_ids(grid_resp_all)
        _render_bulk_update_controls(action_col_all, selected_all, "all")

    else:  # Kurdtel
        try:
            mask = (
                df.get("ticket_group", pd.Series(dtype=str)).astype(str).str.lower().eq("complaints") &
                df.get("complaint_type", pd.Series(dtype=str)).astype(str).str.lower().eq("kurdtel")
            )
            df_kurdtel = df[mask] if not df.empty else df.iloc[0:0]
        except Exception:
            df_kurdtel = df.iloc[0:0]
        df_kurdtel = _apply_ticket_group_filter(df_kurdtel, state_key="ticket_group_filter_kurdtel")
        _status_widget_key = "kurdtel_status_filter_kurdtel"
        _status_sel = st.session_state.get(
            _status_widget_key,
            st.session_state.get("kurdtel_status_filter", "All"),
        )
        st.session_state["kurdtel_status_filter"] = _status_sel
        if (
            _status_sel
            and str(_status_sel).strip().lower() != "all"
            and "complaint_status" in df_kurdtel.columns
        ):
            try:
                normalized_status = str(_status_sel).strip().lower()
                status_series = (
                    df_kurdtel["complaint_status"].astype(str).str.strip().str.lower()
                )
                df_kurdtel = df_kurdtel.loc[status_series == normalized_status]
            except Exception:
                pass
        action_col_kurdtel = _ticket_group_filter_ui(
            "kurdtel",
            df_export=df_kurdtel,
            filename_prefix="kurdtel_tickets",
            show_filter=False,
            status_options=COMP_STATUS,
            status_state_key="kurdtel_status_filter",
            status_label="Complaint Status",
        )
        grid_resp_kurdtel = _render_tickets_grid(
            df_kurdtel,
            ag_key="aggrid_kurdtel",
            width_reference_df=df,
        )
        selected_kurdtel = _extract_selected_ticket_ids(grid_resp_kurdtel)
        _render_bulk_update_controls(action_col_kurdtel, selected_kurdtel, "kurdtel")

    if st.session_state.get("_bulk_update_dialog_open"):
        _bulk_update_ticket_group_dialog()