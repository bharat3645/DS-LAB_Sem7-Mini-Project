#!/bin/bash

# Streamlit Cloud setup script
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
\n\
[theme]\n\
primaryColor = \"#1f77b4\"\n\
backgroundColor = \"#f0f2f6\"\n\
secondaryBackgroundColor = \"#ffffff\"\n\
textColor = \"#1a1a1a\"\n\
font = \"sans serif\"\n\
" > ~/.streamlit/config.toml
