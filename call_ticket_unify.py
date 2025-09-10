import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode, DataReturnMode

st.set_page_config(page_title="CALL_TICKET_UNIFY", layout="wide")

# --- Helper: color missing field labels in red (post-submit) ---
def _color_missing_labels(label_texts):
    if not label_texts:
        return
    sels = []
    for lab in label_texts:
        sels.append(f'label:has(+ div input[aria-label="{lab}"])')
        sels.append(f'label:has(+ div textarea[aria-label="{lab}"])')
        sels.append(f'label:has(+ div [role="combobox"][aria-label="{lab}"])')
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
    "Admin": "build",
    "Exit": "exit_to_app",
}
TAB_NAMES = list(TAB_ICONS.keys())

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

# -------- Top bar --------
html = ['<div class="topbar">']
html.append('<div class="brand">FiberCare</div>')

# Build tab items once for reuse in inline row and hamburger menu
tab_items = []
for name in TAB_NAMES:
    icon = f'<span class="material-icons">{TAB_ICONS[name]}</span>'
    if name == "Call Tickets":
        active_cls = " call-center-active" if st.session_state.active_tab == name else ""
        # Use href navigation so it works without JS
        tab_items.append(
            f'<a href="?tab=Call%20Tickets" target="_self" class="tab{active_cls}">{icon} {name}</a>'
        )
    elif name == "Client":
        active_cls = " call-center-active" if st.session_state.active_tab == name else ""
        tab_items.append(
            f'<a href="?tab=Client" target="_self" class="tab{active_cls}">{icon} {name}</a>'
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
if st.session_state.active_tab == "Call Tickets":
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

def init_ticket_store():
    if "tickets_df" not in st.session_state:
        st.session_state.tickets_df = pd.DataFrame(columns=[
            "id", "created_at", "ticket_group", "ont_id", "type_of_call", "description",
            "activity_inquiry_type", "digicare_issue_type",
            "complaint_type", "refund_type", "online_game", "employee_suggestion", "device_location", "root_cause", "ont_model", "complaint_status", "kurdtel_service_status",
            "osp_type", "city", "issue_type", "fttg", "olt", "second_number", "assigned_to", "address",
            "outage_start_date", "outage_end_date",
            "fttx_job_status", "fttx_job_remarks", "fttx_cancel_reason",
            "callback_status", "callback_reason", "followup_status",
            "reminder_at"
        ])
    if "ticket_seq" not in st.session_state:
        st.session_state.ticket_seq = 1

def add_ticket(row: dict):
    row = row.copy()
    row["id"] = st.session_state.ticket_seq
    row["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.ticket_seq += 1
    st.session_state.tickets_df = pd.concat([st.session_state.tickets_df, pd.DataFrame([row])], ignore_index=True)

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
ASSIGNED_TO            = load_dim_options("cx_dim_assigned_to.xlsx", [])
FTTX_JOB_STATUS        = load_dim_options("cx_dim_fttx_job_status.xlsx", ["In Progress", "Completed", "Cancelled"])
CALLBACK_STATUS        = load_dim_options("cx_dim_callback_status.xlsx", [])
CALLBACK_REASON        = load_dim_options("cx_dim_callback_reason.xlsx", [])
FOLLOWUP_STATUS        = load_dim_options("cx_dim_followup_status.xlsx", [])
DIGICARE_ISSUES        = load_dim_options("cx_dim_digicare_issue.xlsx", [])
ONLINE_GAMES          = load_dim_options("cx_dim_online_game.xlsx", [])
REFUND_TYPES           = load_dim_options("cx_dim_refund_type.xlsx", ["Refund Info", "Refund Request"])

# OSP types (removed "Sub-Districts Interface")
OSP_TYPES = [
    "No Power", "Fiber Cut", "Fast Connector", "Relocate ONT",
    "Degraded", "Rearrange Fiber", "Closure", "Manhole", "Fiber", "Pole"
]

# FTTG fixed options
FTTG_OPTIONS = ["Yes", "No"]

init_ticket_store()

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
    df = st.session_state.tickets_df.copy()
    for k, v in patch.items():
        if k in df.columns:
            df.at[i, k] = v
    st.session_state.tickets_df = df

# ---- New Ticket helpers ----
def _reset_new_ticket_fields():
    """Clear new-ticket form fields and show the new ticket form."""
    st.session_state.show_new_form = True
    for _k in [
        # Activities & Inquiries
        'ont_ai','ont_ai_locked','ai_pending_clear','autofill_message_ai','autofill_level_ai',
        'sb_type_of_activity_inquiries_1','sb_digicare_issue_1',
        # Complaints
        'ont_c','olt_c','olt_c_filled','ont_c_locked','c_pending_clear','autofill_message_c','autofill_level_c',
        'sb_type_of_complaint_1','sb_refund_type_1','sb_complaint_status_1','sb_employee_suggestion_1','sb_root_cause_1','ont_model','sb_device_location_1',
    'assigned_to_c','kurdtel_status_c','online_game_c','online_game_other_c','outage_start_c','outage_end_c',
    'reminder_date_c','reminder_time_c',
        # OSP
        'ont_o','ont_o_locked','osp_pending_clear','autofill_message_o','autofill_level_o','city_o','fttg_o','address_o',
        'sb_osp_appointment_type_1','second_number_o','assigned_to_o',
    ]:
        try:
            st.session_state[_k] = ''
        except Exception:
            pass
    # Ensure date/time widgets start unset to avoid type mismatches
    for _k in ('reminder_date_c','reminder_time_c'):
        try:
            st.session_state.pop(_k, None)
        except Exception:
            pass
    # Ensure Call Type defaults to the first option (e.g., 'Inbound') on fresh form
    try:
        if CALL_TYPES:
            st.session_state['sb_type_of_call_1'] = CALL_TYPES[0]
            st.session_state['sb_type_of_call_2'] = CALL_TYPES[0]
            st.session_state['sb_type_of_call_3'] = CALL_TYPES[0]
        else:
            for _k in ['sb_type_of_call_1','sb_type_of_call_2','sb_type_of_call_3']:
                st.session_state.pop(_k, None)
    except Exception:
        pass

def _is_new_form_dirty() -> bool:
    """Return True if any known new-ticket fields have values indicating in-progress input."""
    candidates = [
        # AI
        'ont_ai','sb_type_of_activity_inquiries_1','sb_digicare_issue_1','sb_type_of_call_1',
        # Complaints
    'ont_c','sb_type_of_complaint_1','sb_refund_type_1','sb_complaint_status_1','sb_type_of_call_2','sb_employee_suggestion_1','sb_root_cause_1','ont_model','sb_device_location_1',
    'assigned_to_c','olt_c','kurdtel_status_c','online_game_c','online_game_other_c','outage_start_c','outage_end_c','ip_c','vlan_c','packet_loss_c','high_ping_c',
    'reminder_date_c','reminder_time_c',
        # OSP
        'ont_o','sb_osp_appointment_type_1','sb_type_of_call_3','second_number_o','assigned_to_o','fttg_o','city_o','address_o',
    ]
    for k in candidates:
        v = st.session_state.get(k)
        # Treat default Call Type (first option) as not-dirty
        if k in ('sb_type_of_call_1','sb_type_of_call_2','sb_type_of_call_3') and CALL_TYPES:
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
        if st.button("Start new ticket", key="confirm_start_new_ticket", use_container_width=True):
            _reset_new_ticket_fields()
            st.session_state['_confirm_new_ticket'] = False
            st.rerun()
    with c2:
        if st.button("Cancel", key="cancel_start_new_ticket", use_container_width=True):
            st.session_state['_confirm_new_ticket'] = False
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
        type_call = st.selectbox("Call Type", CALL_TYPES, index=(CALL_TYPES.index(row["type_of_call"]) if row.get("type_of_call") in CALL_TYPES else None), placeholder="")
        assigned = st.selectbox("Assigned To", ASSIGNED_TO, index=(ASSIGNED_TO.index(row["assigned_to"]) if row.get("assigned_to") in ASSIGNED_TO else None), placeholder="")

        # Group-specific
        updates = {}
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
                        outage_start_e = pd.to_datetime(_s).date() if _s else datetime.now().date()
                    except Exception:
                        outage_start_e = datetime.now().date()
                    try:
                        _e = row.get("outage_end_date")
                        outage_end_e = pd.to_datetime(_e).date() if _e else datetime.now().date()
                    except Exception:
                        outage_end_e = datetime.now().date()
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
                "type_of_call": type_call,
                "assigned_to": assigned,
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
        ("City", "CITY_OPTIONS"),
        ("Issue Type", "ISSUE_TYPES"),
        ("Kurdtel Service Status", "KURDTEL_SERVICE_STATUS"),
        ("Assigned To", "ASSIGNED_TO"),
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
            st.markdown(
                """
                <div class="exp-card">
                  <details open>
                    <summary>#62000 - ccbe.5991.0000 - 1400 - Employee-1 - <span class="dt-green">2026-12-31 23:59:50</span></summary>
                    <div class="cpe-actions">
                      <div class="dropdown" style="position:relative;display:inline-block;">
                        <input type="checkbox" id="nt-dropdown-toggle" class="nt-dropdown-toggle" />
                        <label for="nt-dropdown-toggle" class="cpe-btn cpe-dropdown-btn" aria-haspopup="true"
                          aria-expanded="false"><span class="material-icons">add_ic_call</span> New Call Ticket <span
                            class="material-icons caret">expand_more</span></label>
                        <div class="dropdown-menu"
                          style="position:absolute;top:100%;left:0;background:#fff;border:1px solid #E5E7EB;border-radius:.5rem;box-shadow:0 4px 16px rgba(0,0,0,0.1);min-width:220px;padding:.25rem 0;z-index:10;">
                          <a class="dropdown-item" style="display:block;padding:.5rem .75rem;color:#023058;text-decoration:none;"
                            href="?tab=Call%20Tickets&subtab=Tickets&new=activity&ont=1400" target="_self">Activity / Inquiry</a>
                          <a class="dropdown-item" style="display:block;padding:.5rem .75rem;color:#023058;text-decoration:none;"
                            href="?tab=Call%20Tickets&subtab=Tickets&new=complaint&ont=1400&autofill=1" target="_self">Complaint</a>
                          <a class="dropdown-item" style="display:block;padding:.5rem .75rem;color:#023058;text-decoration:none;"
                            href="?tab=Call%20Tickets&subtab=Tickets&new=osp%20appointment&ont=1400&autofill=1" target="_self">OSP
                            Appointment</a>
                        </div>
                        <label for="nt-dropdown-toggle" class="cpe-dd-overlay" aria-hidden="true"></label>
                      </div>
                      <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">settings_remote</span> Remote
                        Access</span>
                      <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">restart_alt</span> Restart
                        Session</span>
                      <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">remove_circle</span>
                        Unblock</span>
                      <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">description</span>
                        Request</span>
                      <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">public</span> Public
                        Ip</span>
                      <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">credit_card</span>
                        Recharge</span>
                      <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">undo</span> Undo
                        Recharge</span>
                      <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">near_me</span>
                        Transfer</span>
                      <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">find_replace</span>
                        Replace</span>
                      <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">highlight_off</span>
                        Un-Assign</span>
                    </div>
                    <div class="exp-content">
                      <div class="kv-cols">
                        <div class="kv-list">
                          <div class="kv-row">
                            <div class="kv-label">ID</div>
                            <div class="kv-value">62756</div>
                          </div>
                          <div class="kv-sep"></div>
                          <div class="kv-row">
                            <div class="kv-label">Phone</div>
                            <div class="kv-value">07700000000</div>
                          </div>
                          <div class="kv-sep"></div>
                          <div class="kv-row">
                            <div class="kv-label">ONT Model</div>
                            <div class="kv-value">844G</div>
                          </div>
                          <div class="kv-sep"></div>
                          <div class="kv-row">
                            <div class="kv-label">Package</div>
                            <div class="kv-value">Employee-1</div>
                          </div>
                          <div class="kv-sep"></div>
                          <div class="kv-row">
                            <div class="kv-label">Expiration</div>
                            <div class="kv-value">2026-12-31 23:59:50</div>
                          </div>
                        </div>
                        <div class="kv-list">
                          <div class="kv-row">
                            <div class="kv-label">OLT</div>
                            <div class="kv-value">NTWK-Sul-Pasha-OLT-00</div>
                          </div>
                          <div class="kv-sep"></div>
                          <div class="kv-row">
                            <div class="kv-label">ONT ID</div>
                            <div class="kv-value">1400</div>
                          </div>
                          <div class="kv-sep"></div>
                          <div class="kv-row">
                            <div class="kv-label">Serial</div>
                            <div class="kv-value">CXNK00000000</div>
                          </div>
                          <div class="kv-sep"></div>
                          <div class="kv-row">
                            <div class="kv-label">MAC</div>
                            <div class="kv-value">ccbe.5991.0000</div>
                          </div>
                        </div>
                        <div class="kv-list">
                          <div class="kv-row">
                            <div class="kv-label">Operational Status</div>
                            <div class="kv-value">enable</div>
                          </div>
                          <div class="kv-sep"></div>
                          <div class="kv-row">
                            <div class="kv-label">Status</div>
                            <div class="kv-value">Online</div>
                          </div>
                          <div class="kv-sep"></div>
                          <div class="kv-row">
                            <div class="kv-label">IP</div>
                            <div class="kv-value">10.49.72.000</div>
                          </div>
                          <div class="kv-sep"></div>
                          <div class="kv-row">
                            <div class="kv-label">VLAN</div>
                            <div class="kv-value">3021</div>
                          </div>
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
    # Header row: only New Ticket button (single block to avoid extra vertical spacing on mobile)
    # Wrap the button so CSS can target it without :has or text matching
    st.markdown('<div id="new-ticket-btn">', unsafe_allow_html=True)
    new_clicked = st.button("New Call Ticket", key="new_ticket_btn", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # New Ticket form
    if new_clicked:
        if _is_new_form_dirty():
            st.session_state['_confirm_new_ticket'] = True
            _confirm_new_ticket_dialog()
        else:
            _reset_new_ticket_fields()

    if st.session_state.get("show_new_form", False):
        st.markdown('<h3 class="add-new-ticket-h">Add New Ticket</h3>', unsafe_allow_html=True)
    # initialize message keys if missing
        for _k in ['autofill_message_ai','autofill_level_ai','autofill_message_c','autofill_level_c','autofill_message_o','autofill_level_o']:
            if _k not in st.session_state:
                st.session_state[_k] = ''

        # Create tabs (always define before use); allow preselecting Complaints via deep link
        _tabs_order = ["Activities & Inquiries", "Complaints", "OSP Appointments"]
        _focus = st.session_state.get("_new_ticket_focus")
        if _focus in _tabs_order:
            # rotate so focus tab is first
            i = _tabs_order.index(_focus)
            _tabs_order = _tabs_order[i:] + _tabs_order[:i]
        tabs = st.tabs(_tabs_order)
        tabs_by_label = {label: tabs[i] for i, label in enumerate(_tabs_order)}

        # ---- Activities & Inquiries ----
        with tabs_by_label["Activities & Inquiries"]:
            a0c1, a0c2, a0c3 = st.columns(3)
            with a0c1:
                st.selectbox("Activity / Inquiry Type", ACTIVITY_TYPES, index=None, placeholder="", key="sb_type_of_activity_inquiries_1")
            with a0c2: st.empty()
            with a0c3: st.empty()

            with st.form("form_ai", clear_on_submit=False):
                # Show AI toast just above first row (ONT ID)
                _msg = st.session_state.get('autofill_message_ai')
                _lvl = st.session_state.get('autofill_level_ai')
                if _msg:
                    (st.warning if _lvl == 'warning' else st.info)(_msg)
                if st.session_state.get("ai_pending_clear"):
                    st.session_state["ont_ai"] = ""
                    st.session_state["ont_ai_locked"] = ""
                    st.session_state["ai_pending_clear"] = False

                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    st.text_input("ONT ID", key="ont_ai", placeholder="Enter ONT ID")
                with ac2:
                    call_type = st.selectbox("Call Type", CALL_TYPES, index=(0 if CALL_TYPES else None), placeholder="", key="sb_type_of_call_1")
                with ac3:
                    # Assigned To is hidden during creation; auto-assign to current user on save
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

                # Description is only needed for certain activity types
                show_desc = (activity_choice in ("Faulty Device & Adapter", "Information"))
                if show_desc:
                    description = st.text_area("Description", height=100, placeholder="Enter details…")
                else:
                    description = ""

                save_ai = st.form_submit_button("Save Activities & Inquiries")
                if save_ai:
                    activity_type_val = st.session_state.get("sb_type_of_activity_inquiries_1")
                    # Auto-assign to current user (hidden during creation)
                    assigned_ai = "Dahi Nemutlu"
                    missing = []
                    if not st.session_state.get("ont_ai", "").strip():
                        missing.append("ONT ID")
                    if not activity_type_val:
                        missing.append("Activity / Inquiry Type")
                    if not call_type:
                        missing.append("Call Type")
                    # Require Description only for specific activity types
                    if (activity_type_val in ("Faulty Device & Adapter", "Information")) and (not str(description).strip()):
                        missing.append("Description")
                    if missing:
                        _color_missing_labels(missing)
                        st.error("Please fill required fields: " + ", ".join(missing))
                    else:
                        add_ticket({
                                "ticket_group": "Activities & Inquiries",
                                "ont_id": st.session_state["ont_ai"],
                                "type_of_call": call_type,
                                "description": description,
                                "activity_inquiry_type": activity_type_val,
                                "digicare_issue_type": st.session_state.get("sb_digicare_issue_1", "") if activity_type_val == "iQ Digicare" else "",
                                "assigned_to": assigned_ai,
                            })
                        st.success("Activities & Inquiries ticket added.")
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
                    st.session_state["c_pending_clear"] = False

                r1c1, r1c2, r1c3 = st.columns(3)
                with r1c1:
                    locked_c = bool(st.session_state.get("ont_c_locked"))
                    sub_l, sub_r = st.columns([4,1])
                    with sub_l:
                        st.text_input("ONT ID", key="ont_c", placeholder="Enter ONT ID", disabled=locked_c)
                    with sub_r:
                        st.markdown("<div style='height:1.9rem;'></div>", unsafe_allow_html=True)
                        if not locked_c:
                            c_autofill = st.form_submit_button("🔍︎", help="Fetch and autofill related details", use_container_width=True)
                        else:
                            c_remove = st.form_submit_button("❌︎", use_container_width=True)
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
                        st.session_state["olt_c_filled"] = True
                        st.session_state["ont_c_locked"] = True
                        st.session_state["autofill_message_c"] = "Fields autofilled."
                        st.session_state["autofill_level_c"] = "info"
                        st.rerun()
                # Row 1: c2 = Call Type, c3 = Employee Suggestion
                with r1c2:
                    type_of_call_c = st.selectbox("Call Type", CALL_TYPES, index=(0 if CALL_TYPES else None), placeholder="", key="sb_type_of_call_2")
                with r1c3:
                    employee_suggestion = st.selectbox("Employee Suggestion", EMP_SUGGESTION, index=None, placeholder="", key="sb_employee_suggestion_1")
                if not locked_c:
                    if "c_autofill" in locals() and c_autofill:
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
                    if "c_remove" in locals() and c_remove:
                        st.session_state["c_pending_clear"] = True
                        st.session_state["autofill_message_c"] = "Cleared autofill."
                        st.session_state["autofill_level_c"] = "info"
                        st.rerun()

                # Complaint Status is controlled above the form
                # (r0c2 outside the form)

                r2c1, r2c2, r2c3 = st.columns(3)
                with r2c1:
                    root_cause = st.selectbox("Root Cause", ROOT_CAUSE, index=None, placeholder="", key="sb_root_cause_1")
                with r2c2:
                    ont_model = st.selectbox("ONT Model", ONT_MODELS, index=None, placeholder="Click 🔍︎ to auto fill", key="ont_model")
                with r2c3:
                    if st.session_state.get("sb_type_of_complaint_1") == "Kurdtel":
                        # Show Kurdtel Service Status here (read-only) instead of row 5
                        st.selectbox(
                            "Kurdtel Service Status",
                            KURDTEL_SERVICE_STATUS, index=None, placeholder="",
                            key="kurdtel_status_c", disabled=True
                        )
                    else:
                        device_location = st.selectbox("Device Location", DEVICE_LOC, index=None, placeholder="", key="sb_device_location_1")

                # r3: OLT, IP (disabled), VLAN (disabled)
                r3c1, r3c2, r3c3 = st.columns(3)
                with r3c1:
                    olt_c_val = st.text_input("OLT", key="olt_c", placeholder="Click 🔍︎ to auto fill")
                with r3c2:
                    st.text_input("IP", key="ip_c", placeholder="Click 🔍︎ to auto fill", disabled=True)
                with r3c3:
                    st.text_input("VLAN", key="vlan_c", placeholder="Click 🔍︎ to auto fill", disabled=True)

                # r4: Packet Loss, High Ping, Second Number
                r4c1, r4c2, r4c3 = st.columns(3)
                with r4c1:
                    packet_loss_val = st.text_input("Packet Loss", key="packet_loss_c", placeholder="e.g., 10%")
                with r4c2:
                    high_ping_val = st.text_input("High Ping", key="high_ping_c", placeholder="e.g., 180 ms")
                with r4c3:
                    # Show/require Second Number only for specific statuses or complaint types
                    _ct_choice = st.session_state.get("sb_type_of_complaint_1")
                    _status_choice = st.session_state.get("sb_complaint_status_1")
                    _status_norm = (str(_status_choice).strip().lower() if _status_choice else "")
                    _ct_norm = (str(_ct_choice).strip().lower() if _ct_choice else "")
                    _show_second = bool((_status_norm in {"not solved", "pending"}) or (_ct_norm == "problem arising from the extender"))
                    if _show_second:
                        second_number = st.text_input("Second Number")
                    else:
                        st.empty()
                        second_number = ""

                # r5: Outage Start/Kurdtel/Online Game | Outage End/Other (Online Game)
                # initialize outage variables so they're in scope for save handler
                outage_start = None
                outage_end = None
                r5c1, r5c2, r5c3 = st.columns(3)
                with r5c1:
                    # For Online Game Issue show Online Game controls; for Refund Request show outage dates
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
                            .stApp:has(.st-key-online_game_c div[value="Other"]) .st-key-online_game_other_c { display: block !important; }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif (str(st.session_state.get("sb_type_of_complaint_1") or "").strip().lower() == "refund") and (str(st.session_state.get("sb_refund_type_1") or "").strip().lower() == "refund request"):
                        # Ensure prior clears didn't leave an invalid default (e.g., empty string)
                        if isinstance(st.session_state.get("outage_start_c", None), str):
                            st.session_state.pop("outage_start_c", None)
                        outage_start = st.date_input("Outage Start Date", key="outage_start_c")
                    else:
                        st.empty()
                with r5c2:
                    if st.session_state.get("sb_type_of_complaint_1") == "Online Game Issue":
                        st.text_input("Other (Online Game)", key="online_game_other_c", placeholder="Enter game title")
                    elif (str(st.session_state.get("sb_type_of_complaint_1") or "").strip().lower() == "refund") and (str(st.session_state.get("sb_refund_type_1") or "").strip().lower() == "refund request"):
                        if isinstance(st.session_state.get("outage_end_c", None), str):
                            st.session_state.pop("outage_end_c", None)
                        outage_end = st.date_input("Outage End Date", key="outage_end_c")
                    else:
                        st.empty()
                with r5c3:
                    st.empty()
                # Disabled Call-Back / Follow-Up controls during creation
                r6c1, r6c2, r6c3 = st.columns(3)
                with r6c1:
                    st.selectbox(
                        "Call-Back Status", CALLBACK_STATUS,
                        index=None, placeholder="", key="callback_status_c", disabled=True
                    )
                with r6c2:
                    st.selectbox(
                        "Call-Back Reason", CALLBACK_REASON,
                        index=None, placeholder="", key="callback_reason_c", disabled=True
                    )
                with r6c3:
                    st.selectbox(
                        "Follow-Up Status", FOLLOWUP_STATUS,
                        index=None, placeholder="", key="followup_status_c", disabled=True
                    )

                # r7: Assigned To (single)
                # Initialize reminder state as optional (empty by default)
                if "reminder_date_c" not in st.session_state:
                    st.session_state["reminder_date_c"] = None
                if "reminder_time_c" not in st.session_state:
                    st.session_state["reminder_time_c"] = None

                r7c1, r7c2, r7c3 = st.columns(3)
                with r7c1:
                    assigned_to_c = st.selectbox("Assigned To", ASSIGNED_TO, index=None, placeholder="", key="assigned_to_c")
                # Always render reminder inputs; toggle visibility with CSS like Online Game 'Other'
                with r7c2:
                    st.date_input("Reminder Date", key="reminder_date_c")
                    # Render a hidden marker when a date is selected; CSS will use this to reveal the time input
                    if st.session_state.get("reminder_date_c"):
                        st.markdown('<span class="reminder-date-has-value" style="display:none">has-date</span>', unsafe_allow_html=True)
                with r7c3:
                    # Prefill time to current time when a date is picked and time isn't set yet
                    try:
                        from datetime import datetime as _dt
                        if st.session_state.get("reminder_date_c") and not st.session_state.get("reminder_time_c"):
                            st.session_state["reminder_time_c"] = _dt.now().time().replace(second=0, microsecond=0)
                    except Exception:
                        pass
                    st.time_input("Reminder Time", key="reminder_time_c")
                # CSS: hide date/time by default; show date when self-assigned; show time when the date input has a value (pure CSS, like Online Game 'Other')
                st.markdown(
                    """
                    <style>
                    .st-key-reminder_date_c, .st-key-reminder_time_c { display: none !important; }
                    .stApp:has(.st-key-assigned_to_c div[value=\"Dahi Nemutlu\"]) .st-key-reminder_date_c { display: block !important; }
                    /* Show time when the Reminder Date input has a non-empty value (no rerun needed) */
                    .stApp:has(.st-key-assigned_to_c div[value=\"Dahi Nemutlu\"]) 
                          :has(.st-key-reminder_date_c input[value]:not([value=\"\"])) .st-key-reminder_time_c { display: block !important; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                description_c = st.text_area("Description", height=100, placeholder="Describe the complaint…")

                save_c = st.form_submit_button("Save Complaint")
                if save_c:
                    missing = []

                    ont_val = st.session_state.get("ont_c", "").strip()
                    ct_val = st.session_state.get("sb_type_of_complaint_1")
                    es_val = employee_suggestion if "employee_suggestion" in locals() else None
                    rc_val = root_cause if "root_cause" in locals() else None
                    om_val = ont_model if "ont_model" in locals() else None
                    dl_val = device_location if "device_location" in locals() else None
                    cs_val = st.session_state.get("sb_complaint_status_1")
                    tc_val = type_of_call_c if "type_of_call_c" in locals() else None
                    olt_val = st.session_state.get("olt_c", "").strip()
                    sn_val = str(second_number).strip() if "second_number" in locals() else ""
                    desc_val = str(description_c).strip() if "description_c" in locals() else ""
                    ks_val = st.session_state.get("kurdtel_status_c", "").strip()
                    online_game_val = str(st.session_state.get("online_game_c") or "").strip()
                    online_game_other = str(st.session_state.get("online_game_other_c") or "").strip()
                    refund_type_val = str(st.session_state.get("sb_refund_type_1") or "").strip()
                    assigned_c = st.session_state.get("assigned_to_c") or (ASSIGNED_TO[0] if ASSIGNED_TO else "")
                    # Reminder fields (only when self-assigned)
                    current_agent = "Dahi Nemutlu"
                    _self_assigned = str(assigned_c or "").strip() == current_agent
                    rem_date = st.session_state.get("reminder_date_c")
                    rem_time = st.session_state.get("reminder_time_c")
                    reminder_at_str = ""

                    if not ct_val: missing.append("Complaint Type")
                    if not ont_val: missing.append("ONT ID")
                    if not es_val: missing.append("Employee Suggestion")
                    if not rc_val: missing.append("Root Cause")
                    if not om_val: missing.append("ONT Model")
                    if ct_val != "Kurdtel" and not dl_val: missing.append("Device Location")
                    if not cs_val: missing.append("Complaint Status")
                    if not tc_val: missing.append("Call Type")
                    if not olt_val: missing.append("OLT")
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

                    # Validate reminder when self-assigned (optional):
                    # - If a date is set, require time; if both set, ensure future and combine
                    if _self_assigned:
                        if rem_date and not rem_time:
                            missing.append("Reminder Time")
                        elif rem_date and rem_time:
                            try:
                                from datetime import datetime as _dt
                                _dt_combined = _dt.combine(rem_date, rem_time)
                                if _dt_combined <= _dt.now():
                                    missing.append("Reminder Date/Time (future)")
                                else:
                                    reminder_at_str = _dt_combined.isoformat()
                            except Exception:
                                missing.append("Reminder Date/Time")

                    if missing:
                        _color_missing_labels(missing)
                        st.error("Please fill in all required fields: " + ", ".join(missing))
                    else:
                        add_ticket({
                            "ticket_group": "Complaints",
                            "ont_id": ont_val,
                            "type_of_call": tc_val,
                            "description": desc_val,
                            "complaint_type": ct_val,
                            "refund_type": refund_type_val if ct_val == "Refund" else "",
                            "employee_suggestion": es_val,
                            "device_location": (dl_val if ct_val != "Kurdtel" else ""),
                            "root_cause": rc_val,
                            "ont_model": om_val,
                            "complaint_status": cs_val,
                            "kurdtel_service_status": ks_val if ct_val == "Kurdtel" else "",
                            "online_game": ((online_game_val == "Other") and online_game_other or online_game_val) if ct_val == "Online Game Issue" else "",
                            "olt": olt_val,
                            "second_number": sn_val,
                            "assigned_to": assigned_c,
                            "outage_start_date": (outage_start.isoformat() if hasattr(outage_start, 'isoformat') else (str(outage_start) if outage_start else "")),
                            "outage_end_date": (outage_end.isoformat() if hasattr(outage_end, 'isoformat') else (str(outage_end) if outage_end else "")),
                            "ip": st.session_state.get("ip_c", ""),
                            "vlan": st.session_state.get("vlan_c", ""),
                            "packet_loss": str(packet_loss_val or "").strip(),
                            "high_ping": str(high_ping_val or "").strip(),
                            "callback_status": (st.session_state.get("callback_status_c") or ""),
                            "callback_reason": (st.session_state.get("callback_reason_c") or ""),
                            "followup_status": (st.session_state.get("followup_status_c") or "")
                            ,
                            "reminder_at": reminder_at_str
                        })
                        st.success("Complaint ticket added.")
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
    if 'tabs_by_label' in locals():
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
                    st.session_state["osp_pending_clear"] = False

                or1c1, or1c2, or1c3 = st.columns(3)
                with or1c1:
                    locked_o = bool(st.session_state.get("ont_o_locked"))
                    sub_l, sub_r = st.columns([4,1])
                    with sub_l:
                        st.text_input("ONT ID", key="ont_o", placeholder="Enter ONT ID", disabled=locked_o)
                    with sub_r:
                        st.markdown("<div style='height:1.9rem;'></div>", unsafe_allow_html=True)
                        if not locked_o:
                            o_autofill = st.form_submit_button("🔍︎", help="Fetch and autofill related details", use_container_width=True)
                        else:
                            o_remove = st.form_submit_button("❌︎", use_container_width=True)
                # Handle deferred deep-link autofill once for OSP
                if st.session_state.pop("_do_autofill_o", False):
                    if st.session_state.get("ont_o", "").strip():
                        try:
                            st.session_state["city_o"] = CITY_OPTIONS[0] if CITY_OPTIONS else ""
                        except Exception:
                            st.session_state["city_o"] = st.session_state.get("city_o", "")
                        st.session_state["fttg_o"] = "No"
                        st.session_state["address_o"] = "Rizgary Quarter 412, Building 64, Sulaymaniyah, Kurdistan Region"
                        st.session_state["ont_o_locked"] = True
                        st.session_state["autofill_message_o"] = "Fields autofilled."
                        st.session_state["autofill_level_o"] = "info"
                        st.rerun()

                with or1c2:
                    call_type_o = st.selectbox("Call Type", CALL_TYPES, index=(0 if CALL_TYPES else None), placeholder="", key="sb_type_of_call_3")
                with or1c3:
                    second_number_o = st.text_input("Second Number")

                if not locked_o:
                    if "o_autofill" in locals() and o_autofill:
                        if st.session_state.get("ont_o", "").strip():
                            try:
                                st.session_state["city_o"] = CITY_OPTIONS[0] if CITY_OPTIONS else ""
                            except Exception:
                                st.session_state["city_o"] = ""
                            st.session_state["fttg_o"] = "No"
                            st.session_state["address_o"] = "Rizgary Quarter 412, Building 64, Sulaymaniyah, Kurdistan Region"
                            st.session_state["ont_o_locked"] = True
                            st.session_state["autofill_message_o"] = "Fields autofilled."
                            st.session_state["autofill_level_o"] = "info"
                            st.rerun()
                        else:
                            st.session_state["autofill_message_o"] = "Enter ONT ID first."
                            st.session_state["autofill_level_o"] = "warning"
                            st.rerun()
                else:
                    if "o_remove" in locals() and o_remove:
                        st.session_state["osp_pending_clear"] = True
                        st.session_state["autofill_message_o"] = "Cleared autofill."
                        st.session_state["autofill_level_o"] = "info"
                        st.rerun()

                # Row 2: Assigned To (col1), Issue Type (col2), FTTG (col3)
                or2c1, or2c2, or2c3 = st.columns(3)
                with or2c1:
                    assigned_to_o = st.selectbox("Assigned To", ASSIGNED_TO, index=None, placeholder="", key="assigned_to_o")
                with or2c2:
                    issue_type_o = st.selectbox(
                        "Issue Type",
                        ISSUE_TYPES, index=None, placeholder=""
                    )
                with or2c3:
                    fttg_val = st.selectbox(
                        "FTTG",
                        FTTG_OPTIONS, index=None, placeholder="",
                        key="fttg_o", disabled=True
                    )

                # Row 3: City in first column
                or3c1, or3c2, or3c3 = st.columns(3)
                with or3c1:
                    city_val = st.selectbox(
                        "City",
                        CITY_OPTIONS, index=None, placeholder="",
                        key="city_o", disabled=True
                    )
                with or3c2:
                    st.empty()
                with or3c3:
                    st.empty()

                # Row 4: Address (full width)
                address_o = st.text_area("Address", height=80, placeholder="Click 🔍︎ to auto fill", key="address_o")

                # Row 5: Description (full width)
                description_o = st.text_area("Description", height=100, placeholder="Describe the appointment…")

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
                    assigned_o = st.session_state.get("assigned_to_o") or (ASSIGNED_TO[0] if ASSIGNED_TO else "")

                    if not osp_val:  missing.append("OSP Appointment Type")
                    if not ont_val:  missing.append("ONT ID")
                    if not city_sel: missing.append("City")
                    if not sn_val:   missing.append("Second Number")
                    if not issue_sel:missing.append("Issue Type")
                    if not call_sel: missing.append("Call Type")
                    if not fttg_sel: missing.append("FTTG")
                    if not desc_val: missing.append("Description")
                    if not address_val: missing.append("Address")

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
                            "type_of_call": call_sel,
                            "fttg": fttg_sel,
                            "description": desc_val,
                            "address": address_val,
                            "assigned_to": assigned_o,
                        })
                        st.success("Complaint ticket added.")
                        st.session_state.show_new_form = False
                        st.rerun()
            # Developer notes: render OSP appointments markdown below the form inside an expander
            try:
                _dev_md_o = load_implementation_note("osp_appointments.md")
                if _dev_md_o:
                    with st.expander("Implementation notes (OSP Appointments)", expanded=False):
                        st.markdown('<div class="implementation-notes-tag"></div>', unsafe_allow_html=True)
                        st.markdown(_dev_md_o)
            except Exception:
                pass
# ---------- Tickets table (only render in Call Tickets > Tickets) ----------
if st.session_state.active_tab == "Call Tickets" and st.session_state.active_subtab == "Tickets":
    df = st.session_state.tickets_df.copy()

    st.markdown("### Tickets")
    search = st.text_input("Search", placeholder="Search ONT / Description…", label_visibility="collapsed")

    if search:
        s = search.lower()
        def row_match(row):
            fields = [
                "ont_id", "description", "complaint_type", "activity_inquiry_type",
                "kurdtel_service_status", "online_game",
                "osp_type", "second_number", "olt", "city", "issue_type", "fttg", "assigned_to", "address"
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
        from pathlib import Path
        import base64
        from datetime import datetime
    # 3-column row; filter in first column, export icon in third
        c1, c2, c3 = st.columns(3)
        with c1:
            if show_filter:
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
                # Complaint Status filter (used in Kurdtel tab only)
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
                # keep layout spacing without rendering the filter
                st.write("")
        with c3:
            if df_export is not None:
                # Build bytes for download
                data, mime, ext = _export_payload(df_export)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{filename_prefix}_{ts}.{ext}"
                # Load provided Excel.svg and embed as data URI
                try:
                    svg_path = Path(__file__).with_name("assets").joinpath("Excel.svg")
                    if not svg_path.exists():
                        raise FileNotFoundError("assets/Excel.svg not found")
                    svg_bytes = svg_path.read_bytes()
                    svg_b64 = base64.b64encode(svg_bytes).decode("ascii")
                    img_src = f"data:image/svg+xml;base64,{svg_b64}"
                except Exception:
                    img_src = None  # fallback to emoji
                # Build download link with icon only (no tooltip chip)
                card_style = "display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border:1px solid #cfd8dc;border-radius:8px;background:#ffffff;"
                container_style = "display:flex;align-items:center;justify-content:flex-end;height:64px;"
                if img_src:
                    link_html = f"""
                        <div style='{container_style}'>
                            <a href='data:{mime};base64,{base64.b64encode(data).decode('ascii')}' download='{fname}' style='{card_style}'>
                                <img src='{img_src}' alt='Export' style='width:22px;height:22px;display:block;' />
                            </a>
                        </div>
                    """
                else:
                    link_html = f"""
                        <div style='{container_style}'>
                            <a href='data:{mime};base64,{base64.b64encode(data).decode('ascii')}' download='{fname}' style='{card_style};text-decoration:none;'>📥</a>
                        </div>
                    """
                st.markdown(link_html, unsafe_allow_html=True)

    def _apply_ticket_group_filter(df_in: pd.DataFrame) -> pd.DataFrame:
        try:
            sel = st.session_state.get("ticket_group_filter", "All")
            if sel and sel != "All" and "ticket_group" in df_in.columns:
                return df_in[df_in["ticket_group"].astype(str) == sel]
        except Exception:
            pass
        return df_in


def _render_tickets_grid(df_input, ag_key: str | None = None):
    # Build display columns, keep same ordering as before
    display_cols = [
        "id", "created_at", "ticket_group", "ont_id", "type_of_call",
    "activity_inquiry_type", "complaint_type", "refund_type", "online_game", "employee_suggestion",
        "device_location", "root_cause", "ont_model", "complaint_status", "callback_status", "callback_reason", "followup_status",
        "kurdtel_service_status",
        "osp_type", "city", "issue_type", "fttg", "olt", "second_number",
    "fttx_job_status", "assigned_to", "fttx_cancel_reason",
    "address", "description", "fttx_job_remarks", "reminder_at"
    ]
    display_cols = [c for c in display_cols if c in df_input.columns]

    # Some data sources (or an empty store) may not have an 'id' column yet.
    # Try to sort by 'id' when present; otherwise fall back to a safe deterministic order
    try:
        if "id" in display_cols:
            df_sorted = df_input[display_cols].sort_values("id", ascending=False).reset_index(drop=True)
        else:
            # ensure we still include an 'id' column for downstream logic
            df_sorted = df_input[display_cols].reset_index(drop=True)
            if "id" not in df_sorted.columns:
                df_sorted.insert(0, "id", pd.Series(dtype=object))
    except KeyError:
        # unexpected missing column; fallback to safe empty frame with expected columns
        df_sorted = df_input.reindex(columns=display_cols).copy()
        if "id" not in df_sorted.columns:
            df_sorted.insert(0, "id", pd.Series(dtype=object))
        df_sorted = df_sorted.reset_index(drop=True)

    # View df + hidden column for inline edit trigger
    df_view = df_sorted.copy()
    if "id" in df_view.columns:
        df_view["id"] = df_view["id"].astype(str)
    df_view.insert(0, "Edit", "")
    if "_edit_request" not in df_view.columns:
        df_view["_edit_request"] = ""

    gb = GridOptionsBuilder.from_dataframe(df_view, enableRowGroup=False, enableValue=False, enablePivot=False)
    # Disable column filters globally (no filter UI on any column)
    gb.configure_default_column(editable=False, resizable=True, filter=False, flex=1)

    # Long text wrap / auto height
    if "description" in df_view.columns:
        gb.configure_column("description", wrapText=True, autoHeight=True)
    if "address" in df_view.columns:
        gb.configure_column("address", wrapText=True, autoHeight=True)

    # Compute data-driven widths so hidden-tab grids get sensible widths before client-side autosize
    def _estimate_col_width(series, min_w=100, per_char=7, base=20, max_w=420):
        try:
            if series is None or series.empty:
                max_len = 0
            else:
                max_len = int(series.dropna().astype(str).map(len).max())
        except Exception:
            max_len = 0
        w = base + per_char * max_len
        if w < min_w:
            return min_w
        if w > max_w:
            return max_w
        return int(w)

    # Prepare a sample frame for width estimation
    try:
        _sample = df_view[display_cols].astype(str) if not df_view.empty else pd.DataFrame({c: pd.Series(dtype=str) for c in display_cols})
    except Exception:
        _sample = pd.DataFrame({c: pd.Series(dtype=str) for c in display_cols})

    # Apply widths (preserve a tighter fixed width for id)
    for _c in display_cols:
        if _c == "id":
            # Use data-driven width for Ticket ID so it auto-adjusts to content
            gw = _estimate_col_width(_sample[_c] if _c in _sample else None, min_w=80, per_char=8, base=10, max_w=200)
            gb.configure_column("id", minWidth=80, width=gw)
            continue
        if _c == "created_at":
            gw = _estimate_col_width(_sample[_c] if _c in _sample else None, min_w=170)
            gb.configure_column("created_at", minWidth=170, width=gw)
            continue
        if _c == "ticket_group":
            gw = _estimate_col_width(_sample[_c] if _c in _sample else None, min_w=150)
            gb.configure_column("ticket_group", minWidth=150, width=gw)
            continue
        # default estimation for other columns
        gw = _estimate_col_width(_sample[_c] if _c in _sample else None, min_w=120)
        gb.configure_column(_c, minWidth=120, width=gw)

    _header_labels = {
        "id": "#",
        "created_at": "Created At",
        "ticket_group": "Ticket Group",
        "ont_id": "ONT ID",
    "type_of_call": "Call Type",
        "activity_inquiry_type": "Type of Activity & Inquiries",
        "complaint_type": "Type of Complaint",
        "employee_suggestion": "Employee Suggestion",
        "device_location": "Device Location",
        "root_cause": "Root Cause",
        "ont_model": "ONT Model",
    "online_game": "Online Game",
        "complaint_status": "Complaint Status",
    "refund_type": "Refund Type",
        "kurdtel_service_status": "Kurdtel Service Status",
        "osp_type": "OSP Appointment Type",
        "city": "City",
        "issue_type": "Issue Type",
        "fttg": "FTTG",
        "olt": "OLT",
        "second_number": "Second Number",
        "assigned_to": "Assigned To",
        "address": "Address",
        "description": "Description",
        "fttx_job_status": "FTTX Job Status",
        "fttx_job_remarks": "FTTX Job Remarks",
        "fttx_cancel_reason": "FTTX Cancel Reason",
        "callback_status": "Call-Back Status",
        "callback_reason": "Call-Back Reason",
        "followup_status": "Follow-Up Status",
    "reminder_at": "Reminder At",
    }

    for _col, _label in _header_labels.items():
        if _col in df_view.columns:
            gb.configure_column(_col, headerName=_label)

    gb.configure_column("_edit_request", hide=True)

    gb.configure_column(
        "Edit",
        headerName="",
        pinned="left",
        width=60,
        filter=False,
        sortable=False,
        suppressMenu=True,
        floatingFilter=False,
        resizable=False,
        cellRenderer=JsCode("""
            class IconCellRenderer {
                init(params){
                    const id = params && params.data ? params.data.id : null;
                    const span = document.createElement('span');
                    span.className = 'edit-icon';
                    span.textContent = '✏️';
                    span.style.cursor = 'pointer';
                    span.addEventListener('click', () => {
                        try {
                            if (params && params.node) {
                                params.node.setDataValue('_edit_request', String(id ?? '') + '|' + String(Date.now()));
                                params.api.dispatchEvent({ type: 'modelUpdated' });
                            }
                        } catch(e) {}
                    });
                    this.eGui = span;
                }
                getGui(){ return this.eGui; }
            }
        """),
    )

    gb.configure_grid_options(
        onFirstDataRendered=JsCode("""
            function(params){
                try { params.api.sizeColumnsToFit(); } catch(e){}
                try { setTimeout(function(){ params.api.resetRowHeights(); }, 0); } catch(e){}
            }
        """),
    )

    grid_options = gb.build()

    grid_resp = AgGrid(
        df_view,
        gridOptions=grid_options,
        key=ag_key,
        height=520,
        fit_columns_on_grid_load=False,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.AS_INPUT,
        allow_unsafe_jscode=True,
        theme="balham",
    )

    # Detect which row asked to edit and open dialog
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
            _candidates = [r for r in rows if str(r.get('_edit_request', '')).strip() not in ('', '0', 'false', 'None')]
            def _token_ts(flag):
                try:
                    s = str(flag)
                    return float(s.split('|', 1)[1]) if '|' in s else 0.0
                except Exception:
                    return 0.0
            req = max(_candidates, key=lambda r: _token_ts(r.get('_edit_request')), default=None)
            if req:
                _flag = str(req.get('_edit_request', '')).strip()
                _rid = req.get('id')
                def _coerce_int(v):
                    try: return int(v)
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
                _tid = _coerce_int(_rid)
                if _tid is not None:
                    if st.session_state.get('_last_edit_token') != _flag:
                        st.session_state['_last_edit_token'] = _flag
                        st.session_state['_edit_open_id'] = _tid
                        edit_ticket_dialog(_tid)
    except Exception as _e:
        import logging as _lg
        _lg.exception("Inline edit open failed: %s", _e)


if st.session_state.active_tab == "Call Tickets" and st.session_state.active_subtab == "Tickets":
    # Render tickets view: use a controlled selector so only the active grid is created.
    assigned_name = "Dahi Nemutlu"
    # Replace the radio selector with Streamlit tabs (visually native)
    tab_my, tab_all, tab_kurdtel = st.tabs(["My Tickets", "All Tickets", "Kurdtel"])
    with tab_my:
        # Filter to tickets assigned to the current user (case-insensitive)
        if "assigned_to" in df.columns:
            df_my = df[df["assigned_to"].astype(str).str.lower() == assigned_name.lower()]
        else:
            df_my = df.iloc[0:0]
        df_my = _apply_ticket_group_filter(df_my)
        _ticket_group_filter_ui("my", df_export=df_my, filename_prefix="my_tickets")
        _render_tickets_grid(df_my, ag_key="aggrid_my")

    with tab_all:
        df_all = _apply_ticket_group_filter(df)
        _ticket_group_filter_ui("all", df_export=df_all, filename_prefix="all_tickets")
        _render_tickets_grid(df_all, ag_key="aggrid_all")

    with tab_kurdtel:
        # Filter to complaint tickets of type Kurdtel
        try:
            mask = (
                df.get("ticket_group", pd.Series(dtype=str)).astype(str).str.lower().eq("complaints") &
                df.get("complaint_type", pd.Series(dtype=str)).astype(str).str.lower().eq("kurdtel")
            )
            df_kurdtel = df[mask] if not df.empty else df.iloc[0:0]
        except Exception:
            df_kurdtel = df.iloc[0:0]
        df_kurdtel = _apply_ticket_group_filter(df_kurdtel)
        # Apply Complaint Status filter (Kurdtel-only)
        _status_sel = st.session_state.get("kurdtel_status_filter", "All")
        if _status_sel and _status_sel != "All" and "complaint_status" in df_kurdtel.columns:
            try:
                df_kurdtel = df_kurdtel[df_kurdtel["complaint_status"].astype(str) == _status_sel]
            except Exception:
                pass
        _ticket_group_filter_ui(
            "kurdtel",
            df_export=df_kurdtel,
            filename_prefix="kurdtel_tickets",
            show_filter=False,
            status_options=COMP_STATUS,
            status_state_key="kurdtel_status_filter",
            status_label="Complaint Status",
        )
        _render_tickets_grid(df_kurdtel, ag_key="aggrid_kurdtel")