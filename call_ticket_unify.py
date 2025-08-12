import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

st.set_page_config(page_title="UI Prototype", layout="wide")

# --- Helper: color missing field labels in red (post-submit) ---
def _color_missing_labels(label_texts):
    if not label_texts:
        return
    sels = []
    for lab in label_texts:
        sels += [
            f'label:has(+ div input[aria-label="{lab}"])',
            f'label:has(+ div textarea[aria-label="{lab}"])',
            f'label:has(+ div [role="combobox"][aria-label="{lab}"])',
        ]
    css = "<style>" + ", ".join(sels) + "{color:#dc2626 !important;font-weight:600!important;}</style>"
    st.markdown(css, unsafe_allow_html=True)

# -------- Tabs and Material Icon names --------
TAB_ICONS = {
    "Home": "home",
    "Digicare Tickets": "report_problem",
    "Call Center Tickets": "call",
    "Requests": "article",
    "Card": "credit_score",
    "Client": "group",
    "CPE": "router",
    "IVR": "support_agent",
    "Settings": "settings",
    "Admin": "build",
    "Exit": "exit_to_app",  # icon only, no text
}
TAB_NAMES = list(TAB_ICONS.keys())

# -------- State / routing --------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Call Center Tickets"

# Honor query param (no deprecated API)
if "tab" in st.query_params:
    t = st.query_params["tab"]
    if t in TAB_NAMES:
        st.session_state.active_tab = t

# Load CSS
with open("app_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------- Top bar (brand left, tabs right) --------
html = ['<div class="topbar">']
html.append('<div class="brand">FiberCare</div>')
html.append('<div class="tabs">')

for name in TAB_NAMES:
    icon = f'<span class="material-icons">{TAB_ICONS[name]}</span>'
    if name == "Call Center Tickets":
        active_cls = " call-center-active" if st.session_state.active_tab == name else ""
        html.append(
            f'<a href="#" class="tab{active_cls}" '
            f'onclick="window.location.search=`?tab=Call%20Center%20Tickets`; return false;">'
            f'{icon} {name}</a>'
        )
    elif name == "Exit":
        html.append(f'<span class="tab-disabled">{icon}</span>')  # icon only
    else:
        active_cls = " active" if st.session_state.active_tab == name else ""
        html.append(f'<span class="tab-disabled{active_cls}">{icon} {name}</span>')

html.append('</div></div>')
st.markdown("".join(html), unsafe_allow_html=True)

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
                logging.warning(f"Failed to load '%s' from '%s': %s. Using fallback list.", filename, p, e)
    if not found_any:
        logging.warning("Could not find '%s' in any search path. Using fallback list.", filename)
    return (default or [])

def init_ticket_store():
    if "tickets_df" not in st.session_state:
        st.session_state.tickets_df = pd.DataFrame(columns=[
            "id", "created_at", "ticket_group", "ont_id", "type_of_call", "description", "activity_enquiry_type", "complaint_type", "employee_suggestion", "device_location", "root_cause", "ont_model", "complaint_status", "kurdtel_service_status", "osp_type", "city", "issue_type", "fttg", "olt", "second_number", "assigned_to", "address"
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
ACTIVITY_TYPES = load_dim_options("cx_dim_activity_enquiry_type.xlsx", ["General Inquiry", "Billing", "Technical", "Follow-up"])
CALL_TYPES     = load_dim_options("cx_dim_call_type.xlsx", ["Inbound", "Outbound", "Callback"])
COMPLAINT_TYPES= load_dim_options("cx_dim_complaint_type.xlsx", ["Billing", "Connectivity", "Speed", "Other"])
EMP_SUGGESTION = load_dim_options("cx_dim_employee_suggestion.xlsx", ["Escalate", "Schedule OSP", "Remote Fix", "Replace ONT"])
DEVICE_LOC     = load_dim_options("cx_dim_device_location.xlsx", ["Living Room", "Bedroom", "Office", "Other"])
ROOT_CAUSE     = load_dim_options("cx_dim_root_cause.xlsx", ["Power", "Fiber Cut", "Config", "Unknown"])
ONT_MODELS     = load_dim_options("cx_dim_ont_model.xlsx", ["ZTE F680", "Huawei HG8245", "Nokia XS-010X-Q"])
COMP_STATUS    = load_dim_options("cx_dim_complaint_status.xlsx", ["Open", "In Progress", "Pending Customer", "Closed"])

# New dropdowns for OSP
CITY_OPTIONS   = load_dim_options("cx_dim_city.xlsx", [])
ISSUE_TYPES    = load_dim_options("cx_dim_issue_type.xlsx", [])

# NEW: Kurdtel service status options
KURDTEL_SERVICE_STATUS = load_dim_options("cx_dim_kurdtel_service_status.xlsx", [])

# Assignees
ASSIGNED_TO    = load_dim_options("cx_dim_assigned_to.xlsx", [])

# OSP types (removed "Sub-Districts Interface")
OSP_TYPES = [
    "No Power", "Fiber Cut", "Fast Connector", "Relocate ONT",
    "Degraded", "Rearrange Fiber", "Closure", "Manhole", "Fiber", "Pole"
]

# FTTG fixed options
FTTG_OPTIONS = ["Yes", "No"]

init_ticket_store()

# ====================== PAGE CONTENT ======================

st.subheader(st.session_state.active_tab)

if st.session_state.active_tab != "Call Center Tickets":
    st.write("Content for this tab will go here…")
else:
    # Header row: only New Ticket button (left-aligned in a 3-col grid)
    with st.container():
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            # Wrap the button so CSS can target it without :has or text matching
            st.markdown('<div id="new-ticket-btn">', unsafe_allow_html=True)
            new_clicked = st.button("✚ New Ticket", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        # c2 and c3 left empty intentionally

    # New Ticket form
    if new_clicked:
        st.session_state.show_new_form = True
        # reset fields + flags for autofill toggle/lock
        for _k in ['ont_ai','ont_c','olt_c',
                   'olt_c_filled',
                   'ont_ai_locked','ont_c_locked','ai_pending_clear','c_pending_clear',
                   'autofill_message_ai','autofill_level_ai','autofill_message_c','autofill_level_c',
                   # OSP fields
                   'ont_o','ont_o_locked','osp_pending_clear','autofill_message_o','autofill_level_o','city_o','fttg_o',
                   # Kurdtel status field
                   'kurdtel_status_c']:
            st.session_state[_k] = ''

    if st.session_state.get("show_new_form", False):
        st.markdown("### Add New Ticket")
        # Fixed placeholder to prevent layout shift + persistent banner from session
        msg_top = st.empty()
        # initialize message keys if missing
        for _k in ['autofill_message_ai','autofill_level_ai','autofill_message_c','autofill_level_c','autofill_message_o','autofill_level_o']:
            if _k not in st.session_state:
                st.session_state[_k] = ''
        if any([st.session_state.get('autofill_message_ai'),
                st.session_state.get('autofill_message_c'),
                st.session_state.get('autofill_message_o')]):
            # prefer the latest non-empty (Complaints > AI > OSP by precedence)
            msg = (st.session_state.get('autofill_message_c') or
                   st.session_state.get('autofill_message_ai') or
                   st.session_state.get('autofill_message_o'))
            lvl = (st.session_state.get('autofill_level_c') or
                   st.session_state.get('autofill_level_ai') or
                   st.session_state.get('autofill_level_o'))
            if lvl == 'warning':
                msg_top.warning(msg)
            else:
                msg_top.info(msg)
        else:
            msg_top.markdown("<div style='min-height:48px'></div>", unsafe_allow_html=True)

        # Create tabs (always define before use)
        tabs = st.tabs(["Activities & Inquiries", "Complaints", "OSP Appointments"])

        # ---- Activities & Inquiries ----
        with tabs[0]:
            a0c1, a0c2, a0c3 = st.columns(3)
            with a0c1:
                st.selectbox("Type of Activity & Inquiries", ACTIVITY_TYPES, index=None, placeholder="", key="sb_type_of_activity_inquiries_1")
            with a0c2: st.empty()
            with a0c3: st.empty()

            with st.form("form_ai", clear_on_submit=False):
                if st.session_state.get("ai_pending_clear"):
                    st.session_state["ont_ai"] = ""
                    st.session_state["ont_ai_locked"] = ""
                    st.session_state["ai_pending_clear"] = False

                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    st.text_input("ONT ID", key="ont_ai", placeholder="Enter ONT ID")
                with ac2:
                    call_type = st.selectbox("Type of Call", CALL_TYPES, index=None, placeholder="", key="sb_type_of_call_1")
                with ac3:
                    assigned_to_ai = st.selectbox("Assigned To", ASSIGNED_TO, index=None, placeholder="", key="assigned_to_ai")

                description = st.text_area("Description", height=100, placeholder="Enter details…")

                save_ai = st.form_submit_button("Save Activities & Inquiries")
                if save_ai:
                    activity_type_val = st.session_state.get("sb_type_of_activity_inquiries_1")
                    assigned_ai = st.session_state.get("assigned_to_ai") or (ASSIGNED_TO[0] if ASSIGNED_TO else "")
                    missing = []
                    if not st.session_state.get("ont_ai", "").strip():
                        missing.append("ONT ID")
                    if not activity_type_val:
                        missing.append("Type of Activity & Inquiries")
                    if not call_type:
                        missing.append("Type of Call")
                    if not description.strip():
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
                            "activity_enquiry_type": activity_type_val,
                            "assigned_to": assigned_ai,
                        })
                        st.success("Activities & Inquiries ticket added.")
                        st.session_state.show_new_form = False
                        st.rerun()

        # ---- Complaints ----
        with tabs[1]:
            r0c1, r0c2, r0c3 = st.columns(3)
            with r0c1:
                st.selectbox("Type of Complaint", COMPLAINT_TYPES, index=None, placeholder="", key="sb_type_of_complaint_1")
            with r0c2: st.empty()
            with r0c3: st.empty()

            with st.form("form_complaint", clear_on_submit=False):
                if st.session_state.get("c_pending_clear"):
                    st.session_state["ont_c"] = ""
                    st.session_state["olt_c"] = ""
                    st.session_state["olt_c_filled"] = ""
                    st.session_state["ont_c_locked"] = ""
                    st.session_state["ont_model"] = ""
                    st.session_state["kurdtel_status_c"] = ""
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
                    if not locked_c:
                        if "c_autofill" in locals() and c_autofill:
                            if st.session_state.get("ont_c", "").strip():
                                st.session_state["olt_c"] = "xxxx-xxx-xxx-xxx-xx"
                                try:
                                    st.session_state["ont_model"] = ONT_MODELS[0] if ONT_MODELS else ""
                                except Exception:
                                    st.session_state["ont_model"] = ""
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

                with r1c2:
                    employee_suggestion = st.selectbox("Employee Suggestion", EMP_SUGGESTION, index=None, placeholder="", key="sb_employee_suggestion_1")
                with r1c3:
                    complaint_status = st.selectbox("Complaint Status", COMP_STATUS, index=None, placeholder="", key="sb_complaint_status_1")

                r2c1, r2c2, r2c3 = st.columns(3)
                with r2c1:
                    root_cause = st.selectbox("Root Cause", ROOT_CAUSE, index=None, placeholder="", key="sb_root_cause_1")
                with r2c2:
                    ont_model = st.selectbox("ONT Model", ONT_MODELS, index=None, placeholder="Click 🔍︎ to auto fill", key="ont_model")
                with r2c3:
                    device_location = st.selectbox("Device Location", DEVICE_LOC, index=None, placeholder="", key="sb_device_location_1")

                r3c1, r3c2, r3c3 = st.columns(3)
                with r3c1:
                    type_of_call_c = st.selectbox("Type of Call", CALL_TYPES, index=None, placeholder="", key="sb_type_of_call_2")
                with r3c2:
                    olt_c_val = st.text_input("OLT", key="olt_c", placeholder="Click 🔍︎ to auto fill")
                with r3c3:
                    second_number = st.text_input("Second Number")

                r4c1, r4c2, r4c3 = st.columns(3)
                with r4c1:
                    assigned_to_c = st.selectbox("Assigned To", ASSIGNED_TO, index=None, placeholder="", key="assigned_to_c")
                with r4c2:
                    if st.session_state.get("sb_type_of_complaint_1") == "Kurdtel":
                        st.selectbox(
                            "Kurdtel Service Status",
                            KURDTEL_SERVICE_STATUS, index=None, placeholder="",
                            key="kurdtel_status_c", disabled=True
                        )
                with r4c3:
                    st.empty()

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
                    cs_val = complaint_status if "complaint_status" in locals() else None
                    tc_val = type_of_call_c if "type_of_call_c" in locals() else None
                    olt_val = st.session_state.get("olt_c", "").strip()
                    sn_val = str(second_number).strip() if "second_number" in locals() else ""
                    desc_val = str(description_c).strip() if "description_c" in locals() else ""
                    ks_val = st.session_state.get("kurdtel_status_c", "").strip()
                    assigned_c = st.session_state.get("assigned_to_c") or (ASSIGNED_TO[0] if ASSIGNED_TO else "")

                    if not ct_val: missing.append("Type of Complaint")
                    if not ont_val: missing.append("ONT ID")
                    if not es_val: missing.append("Employee Suggestion")
                    if not rc_val: missing.append("Root Cause")
                    if not om_val: missing.append("ONT Model")
                    if not dl_val: missing.append("Device Location")
                    if not cs_val: missing.append("Complaint Status")
                    if not tc_val: missing.append("Type of Call")
                    if not olt_val: missing.append("OLT")
                    if not sn_val: missing.append("Second Number")
                    if not desc_val: missing.append("Description")
                    if ct_val == "Kurdtel" and not ks_val:
                        missing.append("Kurdtel Service Status")

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
                            "employee_suggestion": es_val,
                            "device_location": dl_val,
                            "root_cause": rc_val,
                            "ont_model": om_val,
                            "complaint_status": cs_val,
                            "kurdtel_service_status": ks_val if ct_val == "Kurdtel" else "",
                            "olt": olt_val,
                            "second_number": sn_val,
                        })
                        st.success("Complaint ticket added.")
                        st.session_state.show_new_form = False
                        st.rerun()

        # ---- OSP Appointments ----
        with tabs[2]:
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
                with or1c2:
                    call_type_o = st.selectbox("Type of Call", CALL_TYPES, index=None, placeholder="", key="sb_type_of_call_3")
                with or1c3:
                    second_number_o = st.text_input("Second Number")

                if not locked_o:
                    if "o_autofill" in locals() and o_autofill:
                        if st.session_state.get("ont_o", "").strip():
                            try:
                                st.session_state["city_o"] = CITY_OPTIONS[0] if CITY_OPTIONS else ""
                            except Exception:
                                st.session_state["city_o"] = ""
                            st.session_state["fttg_o"] = "Yes"
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

                or2c1, or2c2, or2c3 = st.columns(3)
                with or2c1:
                    issue_type_o = st.selectbox(
                        "Issue Type",
                        ISSUE_TYPES, index=None, placeholder=""
                    )
                with or2c2:
                    fttg_val = st.selectbox(
                        "FTTG",
                        FTTG_OPTIONS, index=None, placeholder="",
                        key="fttg_o", disabled=True
                    )
                with or2c3:
                    city_val = st.selectbox(
                        "City",
                        CITY_OPTIONS, index=None, placeholder="",
                        key="city_o", disabled=True
                    )

                or3c1, or3c2, or3c3 = st.columns(3)
                with or3c1:
                    assigned_to_o = st.selectbox("Assigned To", ASSIGNED_TO, index=None, placeholder="", key="assigned_to_o")
                with or3c2:
                    st.empty()
                with or3c3:
                    st.empty()

                address_o = st.text_area("Address", height=80, placeholder="Click 🔍︎ to auto fill", key="address_o")
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
                    if not call_sel: missing.append("Type of Call")
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
                        st.success("OSP Appointment ticket added.")
                        st.session_state.show_new_form = False
                        st.rerun()

    # ---------- Tickets table ----------
    df = st.session_state.tickets_df.copy()

    st.markdown("### Tickets")
    search = st.text_input("", placeholder="Search ONT / Description…", label_visibility="collapsed")

    if search:
        s = search.lower()
        def row_match(row):
            fields = ["ont_id", "description", "complaint_type", "activity_enquiry_type",
                      "kurdtel_service_status",
                      "osp_type", "second_number", "olt", "city", "issue_type", "fttg", "assigned_to", "address"]
            return any(str(row.get(f, "")).lower().find(s) >= 0 for f in fields)
        df = df[df.apply(row_match, axis=1)]

    display_cols = [
        "id", "created_at", "ticket_group", "ont_id", "type_of_call",
        "activity_enquiry_type", "complaint_type", "employee_suggestion",
        "device_location", "root_cause", "ont_model", "complaint_status",
        "kurdtel_service_status",
        "osp_type", "city", "issue_type", "fttg", "olt", "second_number", "assigned_to", "address", "description"
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(df[display_cols].sort_values("id", ascending=False), use_container_width=True)
