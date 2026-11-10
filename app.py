"""Run locally with: python -m streamlit run app.py"""

import json
from html import escape
from pathlib import Path

import streamlit as st

from demo import live_requirements, load_examples, recorded_example, run_live
from display_labels import condition_label

st.set_page_config(page_title="Scientific Evidence Agent", layout="wide")
stylesheet = Path(__file__).resolve().parent / ".streamlit/style.css"
st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def markup(html):
    """Only pass trusted markup; escape all claim, corpus and tool text."""
    st.markdown(html, unsafe_allow_html=True)


def use_claim_example():
    selected = st.session_state.get("claim_example")
    st.session_state["claim_text"] = (
        "" if selected is None else load_examples()["examples"][selected]["run"]["claim"]
    )


def show_activity(result):
    with st.container(border=True, key="activity_card"):
        st.subheader("Search record")
        st.caption("Queries and abstracts opened, in order.")
        events = result["tool_trace"]
        if not events:
            st.caption("No search or read activity was recorded.")
        for number, event in enumerate(events, 1):
            args = event.get("arguments", {})
            searching = event["tool"] == "search_abstracts"
            label = "Search corpus" if searching else "Read abstract"
            detail = args.get("query", "") if searching else "Document " + str(args.get("document_id", ""))
            markup(
                '<div class="trace-step">'
                f'<span class="trace-number">{number:02d}</span>'
                '<div class="trace-content">'
                f'<div class="trace-label">{label}</div>'
                f'<p>{escape(str(detail))}</p></div></div>'
            )
            hits = event.get("results", []) if searching else []
            if hits:
                with st.expander(f"{len(hits)} ranked results", expanded=False):
                    for hit in hits:
                        st.write(hit.get("title", "Untitled abstract"))
                        st.caption("Document " + str(hit["document_id"]))
            if event.get("status") == "cached":
                st.caption("Previously read in this run.")
            elif event.get("status") != "ok":
                st.caption("Tool status: " + str(event.get("status", "unknown")))


def show_assessment(item, number):
    variants = {
        "SUPPORT": ("support", "Supports claim"),
        "CONTRADICT": ("contradict", "Contradicts claim"),
        "NOT_ENOUGH_INFO": ("insufficient", "Not enough information"),
    }
    variant, label = variants[item["verdict"]]
    with st.container(border=True, key=f"paper_{number}"):
        markup(
            '<div class="paper-topline">'
            f'<span class="document-reference">Abstract {escape(item["document_id"])}</span>'
            f'<span class="verdict {variant}">{label}</span></div>'
            f'<h3 class="paper-title">{escape(item["title"])}</h3>'
        )
        if item["quotes"]:
            markup('<div class="evidence-label">Quoted evidence</div>')
            for quote in item["quotes"]:
                markup(
                    '<blockquote class="evidence-quote">'
                    f'<span class="sentence-id">[{quote["sentence_id"]}]</span>'
                    f'<p>{escape(quote["text"])}</p></blockquote>'
                )
        else:
            markup('<p class="no-evidence">The agent did not identify a sentence that '
                   'supports or contradicts this claim in this abstract.</p>')
        with st.expander("Agent explanation"):
            st.write(item["explanation"])


def show_result(result):
    if result["mode"] == "recorded":
        st.caption("Recorded run · saved evaluation output. No model request was made.")
    else:
        st.caption("Live run · " + result["model"])
    with st.container(key="claim_summary"):
        markup('<div class="section-kicker report-kicker">Claim under review</div>')
        markup(f'<h2 class="claim-heading">{escape(result["claim"])}</h2>')
        st.caption(f"{result['model']} · {result['retriever'].upper()} retrieval · verdicts apply to individual abstracts")

    with st.container(key="run_metrics"):
        columns = st.columns(4)
        columns[0].metric("Searches", sum(event["tool"] == "search_abstracts" for event in result["tool_trace"]))
        columns[1].metric("Abstracts read", f"{result['abstracts_read']} / 3")
        elapsed, cost = result["elapsed_seconds"], result["estimated_cost_usd"]
        columns[2].metric("Recorded duration" if result["mode"] == "recorded" else "Duration",
                          f"{elapsed:.2f} s" if elapsed is not None else "Unavailable")
        columns[3].metric("SDK-estimated cost", f"${cost:.5f}" if cost is not None else "Unavailable")

    evidence, activity = st.columns([1.85, 1], gap="medium")
    with evidence:
        if result["status"] != "ok":
            with st.container(border=True, key="failure_card"):
                st.error("Processing failure — the run did not produce a valid evidence report.")
                st.write("This is not a NOT_ENOUGH_INFO verdict.")
                with st.expander("Failure details"):
                    st.text(result["error"] or "No error detail was recorded.")
        else:
            for number, item in enumerate(result["assessments"]):
                show_assessment(item, number)
    with activity:
        show_activity(result)
        if result.get("source"):
            source = result["source"]
            with st.expander("Run provenance"):
                st.caption(f"Source: {source['path']}")
                st.caption(f"Claim {source['claim_id']} · {condition_label(source['condition'])} · repetition {source['repetition']}")
                st.caption("Timing and cost describe the original run, not this replay.")
        st.download_button("Download report (JSON)", json.dumps(result, indent=2, ensure_ascii=False),
                           file_name="evidence-report.json", mime="application/json", width="stretch")


markup('''
<div class="masthead">
  <h1 class="project-title">Scientific Evidence</h1>
  <span class="corpus-label">SciFact corpus<br>5,183 research abstracts</span>
</div>
''')
markup('<p class="intro">Inspect claims, source quotations and the searches behind each assessment.</p>')
mode = st.radio("Choose a mode", ["Recorded examples", "Try the SciFact corpus"], horizontal=True,
                label_visibility="collapsed", key="workspace_mode")
if st.session_state.get("previous_mode") != mode:
    st.session_state.pop("result", None)
    st.session_state["previous_mode"] = mode

with st.container(border=True, key="input_card"):
    if mode == "Try the SciFact corpus":
        markup('<p class="scope-note">Searches a fixed collection of biomedical abstracts, '
               'not the web. Missing evidence here does not mean a claim is false.</p>')
        missing = live_requirements()
        if missing:
            st.info("Live setup needed. You can explore Recorded examples without an API key or dataset.")
            with st.expander("Setup details"):
                for message in missing:
                    st.caption(message)
        example_names = ["Exercise and cognition", "Vitamin D and multiple sclerosis", "Onchocerciasis treatment"]
        st.selectbox("Start with a SciFact claim", [None, 0, 1, 2],
                     format_func=lambda i: "Write your own claim" if i is None else example_names[i],
                     key="claim_example", on_change=use_claim_example)
        st.caption("Example claims are drawn from SciFact; their inclusion does not imply they are true.")
        with st.form("claim_form", border=False):
            claim = st.text_area("Scientific claim", height=100, max_chars=2000,
                                 key="claim_text", placeholder="Enter a scientific claim to search within this corpus.")
            action, details = st.columns([1, 2], vertical_alignment="center")
            with action:
                submitted = st.form_submit_button("Check evidence", type="primary", disabled=bool(missing))
            with details:
                st.caption("Haiku · BM25 · guidelines enabled · up to 3 abstracts")
        if submitted:
            st.session_state.pop("result", None)
            if not claim.strip():
                st.warning("Enter a scientific claim first.")
            else:
                with st.spinner("Checking the corpus…", show_time=False, width="stretch"):
                    st.session_state["result"] = run_live(claim)
    else:
        examples = load_examples()["examples"]
        picker, action = st.columns([3, 1], vertical_alignment="bottom")
        with picker:
            selection = st.selectbox("Recorded example", range(len(examples)),
                                     format_func=lambda i: examples[i]["name"])
        with action:
            if st.button("View recorded run", type="primary", width="stretch"):
                st.session_state["result"] = recorded_example(selection)
        st.caption("Real saved runs, including insufficient evidence and a processing failure. No API calls.")

if "result" in st.session_state:
    show_result(st.session_state["result"])
else:
    markup('''
    <div class="empty-report">
      <p>The report will show an assessment for each abstract read, its quoted evidence,
      and the complete search record.</p>
    </div>
    ''')

markup('<div class="app-footer">Verdicts apply to individual abstracts, not the literature as a whole. '
       'Reference checks validate citations; they do not establish scientific correctness. '
       '<span class="footer-credit">Built with the Claude Agent SDK.</span></div>')
