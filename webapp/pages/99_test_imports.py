import streamlit as st

# NO PATH SETUP HERE - testing if Streamlit inherits from index.py

st.write("Testing import without path setup...")

try:
    from webapp.utilities.session import get_or_init_user_session
    st.write("✓ Successfully imported webapp.utilities.session.get_or_init_user_session")
    st.write("This means path setup is NOT needed in every page!")
except ImportError as e:
    st.write(f"✗ Import failed: {e}")
    st.write("This means path setup IS needed in every page.")

try:
    from webapp.utilities.configuration import import_options_general
    st.write("✓ Successfully imported webapp.utilities.configuration.import_options_general")
except ImportError as e:
    st.write(f"✗ Import failed: {e}")

st.write("If you see checkmarks above, the path setup is inherited from index.py!")
