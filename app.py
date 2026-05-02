import streamlit as st

st.set_page_config(layout="wide")

# css structural alignment and result outlines
st.markdown("""
<style>
    /* Lower the home page content */
    .home-spacer { margin-top: 15vh; }

    button[kind="tertiary"] p {
        font-size: 28px !important;
        line-height: 1 !important;
    }
    
    /* Disconnected corner box (camera viewfinder style) */
    .corner-box {
        position: relative;
        padding: 20px;
        margin-bottom: 25px;
        background-color: transparent;
    }
    .corner-box::before, .corner-box::after, .corner-box-inner::before, .corner-box-inner::after {
        content: ''; position: absolute; width: 25px; height: 25px;
    }
    .corner-box::before { 
        top: 0; 
        left: 0; 
        border-top: 2px solid rgba(150, 150, 150, 0.5); 
        border-left: 2px solid rgba(150, 150, 150, 0.5); 
    }
    .corner-box::after { 
        bottom: 0; 
        right: 0; 
        border-bottom: 2px solid rgba(150, 150, 150, 0.5); 
        border-right: 2px solid rgba(150, 150, 150, 0.5); 
    }
    .corner-box-inner::before { 
        top: 0; 
        right: 0; 
        border-top: 2px solid rgba(150, 150, 150, 0.5); 
        border-right: 2px solid rgba(150, 150, 150, 0.5); 
    }
    .corner-box-inner::after { 
        bottom: 0; 
        left: 0; 
        border-bottom: 2px solid rgba(150, 150, 150, 0.5); 
        border-left: 2px solid rgba(150, 150, 150, 0.5); 
    }
    
    div[data-baseweb="input"] { background-color: transparent; }
</style>
""", unsafe_allow_html=True)

# STATE MANAGEMENT
if 'view' not in st.session_state:
    st.session_state.view = 'home'
if 'query' not in st.session_state:
    st.session_state.query = ''
if 'page' not in st.session_state:
    st.session_state.page = 1

# initialize widget keys in session state
if 'home_search_box' not in st.session_state:
    st.session_state.home_search_box = ""
if 'top_search_box' not in st.session_state:
    st.session_state.top_search_box = ""

# CALLBACK FUNCTIONS
def go_home():
    st.session_state.view = 'home'
    st.session_state.query = ''
    st.session_state.home_search_box = ''
    st.session_state.page = 1

def clear_top_search():
    st.session_state.top_search_box = ''

def execute_home_search():
    search_term = st.session_state.home_search_box
    if search_term.strip():
        st.session_state.query = search_term
        st.session_state.top_search_box = search_term 
        st.session_state.view = 'results'
        st.session_state.page = 1

def execute_top_search():
    search_term = st.session_state.top_search_box
    if search_term.strip():
        st.session_state.query = search_term
        st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

# =============================================================================
# HOME PAGE
# =============================================================================
if st.session_state.view == 'home':
    st.markdown('<div class="home-spacer"></div>', unsafe_allow_html=True)
    
    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
    
    with center_col:
        # search engine title
        st.markdown("<div style='text-align: center; font-size: 2.5rem; " \
            "font-weight: 700; margin-bottom: 1rem;'>//placeholder</div>", 
            unsafe_allow_html=True)
        
        search_input_col, search_btn_col = st.columns([10, 1])
        
        with search_input_col:
            st.text_input("Search", key="home_search_box", 
                          on_change=execute_home_search, 
                          label_visibility="collapsed", 
                          placeholder="...")
            
        with search_btn_col:
            # remove button outlines
            st.button("⌕", key="home_search_btn", 
                      on_click=execute_home_search, 
                      type="tertiary", use_container_width=True)
        
        # home page search bar subtext
        st.markdown("<div style='text-align: left; font-size: 14px; color: gray; " \
            "margin-top: 5px; margin-left: 5px;'>Financial Literacy for Students</div>", 
            unsafe_allow_html=True)

# =============================================================================
# RESULTS PAGE
# =============================================================================
elif st.session_state.view == 'results':
    
    nav_home, nav_search, nav_spacer = st.columns([0.5, 4, 3.5], 
                                                  vertical_alignment="bottom")
    
    with nav_home:
        st.button("⌂", on_click=go_home, type="tertiary", 
                  use_container_width=True)
        
    with nav_search:
        sc_input, sc_x, sc_mag = st.columns([8, 1, 1])
        with sc_input:
            st.text_input("Search", key="top_search_box", 
                          on_change=execute_top_search, 
                          label_visibility="collapsed")
        with sc_x:
            st.button("✕", on_click=clear_top_search, 
                      type="tertiary", 
                      use_container_width=True)
        with sc_mag:
            st.button("⌕", key="top_search_btn", on_click=execute_top_search, 
                      type="tertiary", use_container_width=True)

    st.divider()

    res_left_spacer, results_col, res_right_spacer = st.columns([1, 3, 1])
    
    with results_col:
        for i in range(1, 6):
            doc_num = i + ((st.session_state.page - 1) * 5)
            st.markdown(f"""
            <div class="corner-box">
                <div class="corner-box-inner"></div>
                <strong>//Doc ID:</strong><br><br>
            //document text 
            </div>
            """, unsafe_allow_html=True)
            
        # PAGINATION
        max_pages = 5

        pag_left, pag_spacer, pag_prev, pag_next = st.columns([2, 6, 1, 1], 
                                                              vertical_alignment="center")
        
        with pag_left:
            st.markdown(f"<div style='text-align: left; color: gray; font-size: 14px; padding-top: 8px;'>Pg. {st.session_state.page}/{max_pages}</div>", 
                        unsafe_allow_html=True)
        with pag_spacer:
            st.empty() 
        with pag_prev:
            st.button("Prev", disabled=(st.session_state.page <= 1), 
                      on_click=prev_page, use_container_width=True)
        with pag_next:
            st.button("Next", disabled=(st.session_state.page >= max_pages), 
                      on_click=next_page, use_container_width=True)