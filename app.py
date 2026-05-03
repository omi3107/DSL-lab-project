"""
app.py — MeetingMind AI Dashboard
Streamlit UI matching the meetingmind.html wireframe layout.
Uses ML classification + Gemini/Groq AI framing pipeline.
"""
import json
import logging
import sys
import io
import os
import uuid
import time
import html as html_mod
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
try:
    import markdown
except ImportError:
    markdown = None
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title='MeetingMind', page_icon='🧠', layout='wide', initial_sidebar_state='collapsed')

# Keep Streamlit chrome out of the way so the embedded UI feels like the original page.
st.markdown(
    """
    <style>
      #MainMenu, footer, header { visibility: hidden; display: none !important; }
      header[data-testid="stHeader"] { display: none !important; }
      .stApp { background: #0c0e14; }
      [data-testid="stSidebar"] { display: none; }
      .block-container { 
          padding-top: 0 !important; 
          padding-bottom: 0 !important; 
          padding-left: 0 !important; 
          padding-right: 0 !important; 
          margin-top: 0 !important;
          max-width: 100% !important; 
      }
      div[data-testid="stVerticalBlock"] > div:first-child {
          padding-top: 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

html_path = Path(__file__).with_name('meetingmind.html')
if not html_path.exists():
    st.error(f'Could not find HTML file: {html_path.name}')
    st.stop()

html = html_path.read_text(encoding='utf-8')
components.html(html, height=650, scrolling=False)

