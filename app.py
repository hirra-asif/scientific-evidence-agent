"""Run locally with: python -m streamlit run app.py"""

import json

import streamlit as st

from demo import live_requirements, load_examples, recorded_example, run_live

st.set_page_config(page_title="Scientific Evidence Agent", page_icon="🔎", layout="wide")
st.markdown("""
<style>
  .stApp { background: #f8f8f4; }
  .block-container { max-width: 1120px; padding-top: 2.3rem; padding-bottom: 3rem; }
  h1 { letter-spacing: -0.045em; font-weight: 650 !important; }
  h2, h3 { letter-spacing: -0.02em; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
  [data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 14px; }
  [data-testid="stCaptionContainer"] { color: #616b68; }
  .stButton button { border-radius: 9px; }
</style>
""", unsafe_allow_html=True)


def show_activity(result):
    with st.expander("Searches and abstracts read", expanded=True):
        events = result["tool_trace"]
        if not events:
            st.caption("No search or read activity was recorded.")
        for number, event in enumerate(events, 1):
            args = event.get("arguments", {})
            if event["tool"] == "search_abstracts":
                st.markdown(f"**{number}. Search**")
                st.text(args.get("query", ""))
                hits = event.get("results", [])
                if hits:
                    st.caption("Top results: " + " · ".join(str(hit["document_id"]) for hit in hits))
            else:
                st.markdown(f"**{number}. Read abstract {args.get('document_id', '')}**")
            if event.get("status") not in ("ok", "cached"):
                st.caption("Tool status: " + str(event.get("status", "unknown")))


def show_result(result):
    st.divider()
    if result["mode"] == "recorded":
        st.info("Recorded run · loaded from the saved evaluation. No model request was made.")
    else:
        st.caption("LIVE RUN · " + result["model"])
    st.subheader("Evidence report")
    st.write(result["claim"])
    st.caption(f"{result['model']} · {result['retriever'].upper()} retrieval · verdicts apply to individual abstracts")
    columns = st.columns(4)
    columns[0].metric("Searches", sum(event["tool"] == "search_abstracts" for event in result["tool_trace"]))
    columns[1].metric("Abstracts read", f"{result['abstracts_read']} / 3")
    elapsed, cost = result["elapsed_seconds"], result["estimated_cost_usd"]
    columns[2].metric("Recorded duration" if result["mode"] == "recorded" else "Duration",
                      f"{elapsed:.2f} s" if elapsed is not None else "Unavailable")
    columns[3].metric("SDK-estimated cost", f"${cost:.5f}" if cost is not None else "Unavailable")
    if result["status"] != "ok":
        st.error("Processing failure — the run did not produce a valid evidence report.")
        st.write("This is not a NOT_ENOUGH_INFO verdict.")
        with st.expander("Failure details"):
            st.text(result["error"] or "No error detail was recorded.")
    else:
        for item in result["assessments"]:
            with st.container(border=True):
                st.caption("ABSTRACT " + item["document_id"])
                st.subheader(item["title"])
                labels = {"SUPPORT": "Supports this claim", "CONTRADICT": "Contradicts this claim",
                          "NOT_ENOUGH_INFO": "Not enough information"}
                st.markdown("**" + labels[item["verdict"]] + "**")
                if item["quotes"]:
                    st.caption("EXACT EVIDENCE · numbered corpus sentences")
                    for quote in item["quotes"]:
                        st.write(f"[{quote['sentence_id']}] {quote['text']}")
                else:
                    st.caption("No evidence sentences cited for this verdict.")
                with st.expander("Agent explanation"):
                    st.write(item["explanation"])
    show_activity(result)
    if result.get("source"):
        source = result["source"]
        st.caption(f"Source: {source['path']} · claim {source['claim_id']} · "
                   f"{source['condition']} · repetition {source['repetition']}")
    st.download_button("Download this report", json.dumps(result, indent=2, ensure_ascii=False),
                       file_name="evidence-report.json", mime="application/json")


st.caption("SCIFACT / CLAUDE AGENT SDK")
st.title("Follow the evidence.")
st.write("Check a scientific claim against 5,183 research abstracts. Inspect the searches, verdicts and exact source sentences.")
mode = st.radio("Choose a mode", ["Check a claim", "Recorded examples"], horizontal=True)
if st.session_state.get("previous_mode") != mode:
    st.session_state.pop("result", None)
    st.session_state["previous_mode"] = mode

with st.container(border=True):
    if mode == "Check a claim":
        st.subheader("What would you like to check?")
        missing = live_requirements()
        if missing:
            st.info("Live setup needed. You can explore Recorded examples without an API key or dataset.")
            for message in missing:
                st.caption(message)
        with st.form("claim_form"):
            claim = st.text_area("Scientific claim", height=110, max_chars=2000,
                                 placeholder="For example: Participating in six months of physical activity improves cognitive functioning.")
            st.caption("Haiku · BM25 · evidence guidelines enabled · up to three abstracts")
            submitted = st.form_submit_button("Check evidence", type="primary", disabled=bool(missing))
        if submitted:
            st.session_state.pop("result", None)
            if not claim.strip():
                st.warning("Enter a scientific claim first.")
            else:
                with st.status("Searching abstracts and assessing the evidence…", expanded=True) as status:
                    st.write("The agent chooses its searches and reads. The completed tool trace appears below.")
                    result = run_live(claim)
                    st.session_state["result"] = result
                    status.update(label="Report ready" if result["status"] == "ok" else "Run could not complete",
                                  state="complete" if result["status"] == "ok" else "error", expanded=False)
    else:
        st.subheader("Explore a recorded run")
        st.caption("Real saved runs: evidence found, insufficient evidence, and a processing failure. Viewing them makes no API calls.")
        examples = load_examples()["examples"]
        selection = st.selectbox("Recorded example", range(len(examples)),
                                 format_func=lambda i: examples[i]["name"])
        if st.button("View recorded run", type="primary"):
            st.session_state["result"] = recorded_example(selection)

if "result" in st.session_state:
    show_result(st.session_state["result"])
else:
    st.caption("Reports show verdicts for individual abstracts, not a consensus across the literature. Reference checks validate citations; they do not establish scientific correctness.")
