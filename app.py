import streamlit as st
import pandas as pd

from src.DataProcessing import DataProcessing
from src.bm25_search import BM25Search
from src.vsm_search import VSMSearch
from src.bim_search import BIMSearch
from src.language_model import LanguageModel

st.set_page_config(layout="wide")

# SEARCH ENGINE INIT (cached so it only builds once)
@st.cache_resource
def load_engines():
    p = DataProcessing()
    return p, {
        'BM25': BM25Search(p.docs, p.qrels),
        'VSM': VSMSearch(p.docs, p.qrels, p.vectorizer, p.tfidf),
        'BIM': BIMSearch(p.docs, p.qrels),
        'LM Unigram': LanguageModel(docs=p.docs, relevance=p.qrels, model='unigram', lambda_=0.3),
        'LM Bigram': LanguageModel(docs=p.docs, relevance=p.qrels, model='bigram', lambda_=0.3),
    }

data, engines = load_engines()

def run_live_search(query_text, engine_name):
    engine = engines[engine_name]
    q_df = pd.DataFrame([{'query_id': 0, 'text': query_text}])
    result = engine.search(q_df).iloc[0]
    return [(data.docs.iloc[i]['doc_id'], data.docs.iloc[i]['text'], s)
            for i, s in zip(result['ranked_indices'], result['scores'])]

# css structural alignment and result outlines
st.markdown("""
<style>
    /* Lower the home page content */
    .home-spacer { margin-top: 15vh; }

    button[kind="tertiary"] p {
        font-size: 24px !important;
        line-height: 1 !important;
    }
    
    div[data-baseweb="input"] { background-color: transparent; }

    /* Hide "Press Enter to apply" hint */
    div[data-testid="InputInstructions"] { 
        display: none !important; 
    }

    /* Green Highlight on Search Box Focus */
    div[data-baseweb="input"]:focus-within {
        border-color: #2e8b57 !important;
    }
    div[data-baseweb="input"]:focus-within > div {
        box-shadow: #2e8b57 0px 0px 0px 1px inset !important;
    }

    button[kind="primary"] {
        position: relative !important;
        padding: 25px !important;
        margin-bottom: 10px !important;
        background-color: transparent !important;
        border: none !important;
        text-align: left !important;
        white-space: normal !important; 
        height: 140px !important; 
        display: block !important;
        width: 100% !important;
        color: inherit !important;
    }
    
    button[kind="primary"] p {
        width: 100% !important;
        text-align: left !important;
        margin: 0 !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 3 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
    }
    
    /* Suppress Streamlit's default red hover effect on primary buttons */
    button[kind="primary"]:hover {
        border: none !important;
        background-color: rgba(150, 150, 150, 0.05) !important;
        color: inherit !important;
    }

    /* Paint the viewfinder corners directly onto the primary button */
    button[kind="primary"]::before, 
    button[kind="primary"]::after, 
    button[kind="primary"] div::before, 
    button[kind="primary"] div::after {
        content: ''; position: absolute; width: 25px; height: 25px; pointer-events: none;
    }
    button[kind="primary"]::before { 
        top: 0; left: 0; 
        border-top: 2px solid rgba(150, 150, 150, 0.5); 
        border-left: 2px solid rgba(150, 150, 150, 0.5); 
    }
    button[kind="primary"]::after { 
        bottom: 0; right: 0; 
        border-bottom: 2px solid rgba(150, 150, 150, 0.5); 
        border-right: 2px solid rgba(150, 150, 150, 0.5); 
    }
    
    button[kind="primary"] div { 
        width: 100% !important; 
        display: flex !important;
        flex-direction: column !important; 
        align-items: flex-start !important; 
        justify-content: flex-start !important;
        text-align: left !important;
    }
    
    button[kind="primary"] div::before { 
        top: 0; right: 0; 
        border-top: 2px solid rgba(150, 150, 150, 0.5); 
        border-right: 2px solid rgba(150, 150, 150, 0.5); 
    }
    button[kind="primary"] div::after { 
        bottom: 0; left: 0; 
        border-bottom: 2px solid rgba(150, 150, 150, 0.5); 
        border-left: 2px solid rgba(150, 150, 150, 0.5); 
    }
</style>
""", unsafe_allow_html=True)

# STATE MANAGEMENT
if 'view' not in st.session_state:
    st.session_state.view = 'home'
if 'query' not in st.session_state:
    st.session_state.query = ''
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'engine' not in st.session_state:
    st.session_state.engine = 'BM25'
if 'results' not in st.session_state:
    st.session_state.results = []
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = {} 

if 'home_search_box' not in st.session_state:
    st.session_state.home_search_box = ""
if 'top_search_box' not in st.session_state:
    st.session_state.top_search_box = st.session_state.query

# CALLBACK FUNCTIONS
def go_home():
    st.session_state.view = 'home'
    st.session_state.query = ''
    st.session_state.home_search_box = ''
    st.session_state.page = 1

def return_to_results():
    st.session_state.view = 'results'
    st.session_state.top_search_box = st.session_state.query

def clear_top_search():
    st.session_state.top_search_box = ''

def execute_home_search():
    search_term = st.session_state.home_search_box
    if search_term.strip():
        st.session_state.query = search_term
        st.session_state.top_search_box = search_term
        st.session_state.view = 'results'
        st.session_state.page = 1
        st.session_state.results = run_live_search(search_term, st.session_state.engine)

def execute_top_search():
    search_term = st.session_state.top_search_box
    if search_term.strip():
        st.session_state.query = search_term
        st.session_state.page = 1
        st.session_state.results = run_live_search(search_term, st.session_state.engine)

def on_engine_change():
    if st.session_state.query:
        st.session_state.results = run_live_search(st.session_state.query, st.session_state.engine)
        st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

def view_document(doc_id, doc_text, score):
    st.session_state.current_doc = {'id': doc_id, 'text': doc_text, 'score': score}
    st.session_state.view = 'document'

# =============================================================================
# HOME PAGE
# =============================================================================
if st.session_state.view == 'home':
    st.markdown('<div class="home-spacer"></div>', unsafe_allow_html=True)
    
    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
    
    with center_col:
        st.markdown("<div style='text-align: center; font-size: 2.5rem; " \
            "font-weight: 700; margin-bottom: 1rem;'>//placeholder</div>", 
            unsafe_allow_html=True)
        
        search_input_col, search_eng_col, search_btn_col = st.columns([8.2, 1.8, 0.8])
        
        with search_input_col:
            st.text_input("Search", key="home_search_box", 
                          on_change=execute_home_search, 
                          label_visibility="collapsed", 
                          placeholder="Search...")
                          
        with search_eng_col:
            st.selectbox("Engine", list(engines.keys()), key="engine", label_visibility="collapsed")
            
        with search_btn_col:
            st.button("🔍", key="home_search_btn", 
                      on_click=execute_home_search, 
                      type="tertiary", use_container_width=True)
        
        st.markdown("<div style='text-align: left; font-size: 14px; color: gray; " \
            "margin-top: 5px; margin-left: 5px;'>Financial Literacy for Students</div>",
            unsafe_allow_html=True)

# =============================================================================
# RESULTS PAGE
# =============================================================================
elif st.session_state.view == 'results':
    
    nav_home, nav_search, nav_spacer = st.columns([0.5, 4, 3.5])
    
    with nav_home:
        st.button("⌂ Home", on_click=go_home, type="tertiary", use_container_width=True)
        
    with nav_search:
        sc_input, sc_eng, sc_x, sc_mag = st.columns([7.5, 1.5, 0.6, 0.6])
        with sc_input:
            st.text_input("Search", key="top_search_box",
                          on_change=execute_top_search,
                          label_visibility="collapsed")
        with sc_eng:
            st.selectbox("Engine", list(engines.keys()), key="engine",
                         on_change=on_engine_change, label_visibility="collapsed")
        with sc_x:
            st.button("✕", on_click=clear_top_search, type="tertiary", use_container_width=True)
        with sc_mag:
            st.button("🔍", key="top_search_btn", on_click=execute_top_search,
                      type="tertiary", use_container_width=True)

    st.divider()

    res_left_spacer, results_col, res_right_spacer = st.columns([1, 3, 1])

    with results_col:
        page_size = 5
        start = (st.session_state.page - 1) * page_size
        end = start + page_size
        page_results = st.session_state.results[start:end]

        for doc_id, doc_text, score in page_results:
            preview = doc_text if len(doc_text) <= 400 else doc_text[:400] + "..."
            
            button_label = f"Doc ID: {doc_id}  |  Score: {score:.4f}\n\n{preview}"
            
            st.button(
                button_label, 
                key=f"doc_{doc_id}", 
                on_click=view_document, 
                args=(doc_id, doc_text, score), 
                type="primary", 
                use_container_width=True
            )
            
        # PAGINATION
        max_pages = 5

        pag_left, pag_spacer, pag_prev, pag_next = st.columns([2, 7, 1, 1])
        
        with pag_left:
            st.markdown(f"<div style='text-align: left; color: gray; font-size: 14px; padding-top: 8px;'>Pg. {st.session_state.page}/{max_pages}</div>", 
                        unsafe_allow_html=True)
        with pag_spacer:
            st.empty() 
        with pag_prev:
            st.button("Prev", disabled=(st.session_state.page <= 1), on_click=prev_page, use_container_width=True)
        with pag_next:
            st.button("Next", disabled=(st.session_state.page >= max_pages), on_click=next_page, use_container_width=True)

# =============================================================================
# FULL DOCUMENT PAGE
# =============================================================================
elif st.session_state.view == 'document':
    st.button("< Back to Results", on_click=return_to_results)
    st.divider()
    
    doc = st.session_state.current_doc
    
    st.subheader(f"Document ID: {doc.get('id')}", anchor=False)
    st.write(doc.get('text'))