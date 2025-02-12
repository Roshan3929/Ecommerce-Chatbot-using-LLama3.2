import streamlit as st
from chatbot import chatbot_page
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2

# Database connection function
def get_db_connection():
    connection = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="Roshan7446",
        port=5432
    )
    return connection

# Function to simulate user authentication
def authenticate(email, password):
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT role FROM customer WHERE email = %s AND password = %s;", (email, password))
            user_role = cur.fetchone()
            cur.close()
            conn.close()
            if user_role:
                return user_role[0]
        except Exception as e:
            print(f"Error during authentication: {e}")
            return None
    return None

# Login/Signup Page
def login_signup_page():
    # Set page layout and style
    st.set_page_config(page_title="Login / Sign Up", layout="centered")
    st.markdown(
        """
        <style>
        .stTextInput input {
            border-radius: 10px;
            padding: 10px;
        }
        .stButton button {
            width: 100%;
            border-radius: 10px;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stMarkdown h1 {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title and description
    st.markdown("<h1>🔐 Login / Sign Up</h1>", unsafe_allow_html=True)
    st.write("Welcome to the Chatbot! Please log in or sign up to continue.")

    # Role selection in the sidebar
    with st.sidebar:
        st.subheader("Customer Login / Sign Up")
        option = st.radio("", ("Login", "Sign Up"))  # Removed "Choose an option"

    # Role selection
    role = st.radio("Select your role:", ("Customer", "System Administrator"))

    if role == "Customer":
        if option == "Login":
            st.write("### Customer Login")
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            if st.button("Login"):
                user_role = authenticate(email, password)
                if user_role == "customer":
                    st.session_state.logged_in = True
                    st.session_state.role = "customer"
                    st.session_state.username = email
                    st.session_state.page = "chatbot"  # Navigate to Chatbot Page
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

        elif option == "Sign Up":
            st.write("### Customer Sign Up")
            name = st.text_input("Name", placeholder="Enter your full name")
            email = st.text_input("Email Address", placeholder="you@example.com")
            phone = st.text_input("Phone Number", placeholder="Enter your phone number")
            new_password = st.text_input("Choose a password", type="password", placeholder="Enter 6 characters or more")
            confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter your password")

            if st.button("Sign Up"):
                if new_password == confirm_password:
                    conn = get_db_connection()
                    if conn is not None:
                        try:
                            cur = conn.cursor()
                            cur.execute("INSERT INTO customer (name, email, phone, password, role) VALUES (%s, %s, %s, %s, %s);",
                                        (name, email, phone, new_password, "customer"))
                            conn.commit()
                            cur.close()
                            conn.close()
                            st.success("Account created successfully! Please log in.")
                        except Exception as e:
                            st.error(f"Error creating account: {e}")
                    else:
                        st.error("Database connection failed.")
                else:
                    st.error("Passwords do not match.")

    elif role == "System Administrator":
        st.subheader("System Administrator Login")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("Login"):
            user_role = authenticate(username, password)
            if user_role == "admin":
                st.session_state.logged_in = True
                st.session_state.role = "admin"
                st.session_state.username = username
                st.session_state.page = "admin"  # Navigate to Admin Page
                st.rerun()
            else:
                st.error("Invalid username or password.")

# Simulated data for the admin page
def load_admin_data():
    # Simulated conversation logs
    conversation_logs = pd.DataFrame({
        "Customer Name": ["Alice", "Bob", "Charlie", "David"],
        "Order ID": ["12345", "67890", "11223", "44556"],
        "Query Type": ["Shipping", "Returns", "Payments", "Tracking"],
        "Date": ["2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04"],
        "Resolution Status": ["Resolved", "Unresolved", "Resolved", "Escalated"],
        "Chat Transcript": [
            "C: Where is my order? B: Your order is out for delivery.",
            "C: I want to return an item. B: Please follow the return process.",
            "C: My payment failed. B: Please try again or use a different card.",
            "C: I can't track my order. B: Let me escalate this issue."
        ]
    })

    # Simulated FAQ data
    faq_data = pd.DataFrame({
        "Question": [
            "How can I track my order?",
            "What is your return policy?",
            "How do I update my payment information?"
        ],
        "Answer": [
            "You can track your order using the tracking number in your confirmation email.",
            "You can return items within 30 days of purchase.",
            "You can update your payment information in the account settings."
        ],
        "Category": ["Shipping", "Returns", "Payments"],
        "Usage Count": [150, 75, 50]
    })

    return conversation_logs, faq_data

# Dashboard Overview
def dashboard_overview():
    st.subheader("📊 Dashboard Overview")
    st.markdown(
        """
        <style>
        .stMetric {
            border: 1px solid #e1e4e8;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background-color: #f9f9f9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Conversations", 1200)
    with col2:
        st.metric("Resolution Rate", "85%")
    with col3:
        st.metric("Average Response Time", "2.5s")
    with col4:
        st.metric("Customer Satisfaction", "4.3/5")
    with col5:
        st.metric("Active Issues", 15)

    # Graphs/Charts
    st.write("### Trends Over Time")
    data = pd.DataFrame({
        "Date": pd.date_range(start="2023-09-01", periods=30, freq="D"),
        "Queries": np.random.randint(50, 200, size=30)
    })
    fig, ax = plt.subplots()
    ax.plot(data["Date"], data["Queries"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Queries")
    st.pyplot(fig)

# Analytics and Reporting
def analytics_and_reporting():
    st.subheader("📈 Analytics and Reporting")
    st.write("### Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Response Time", "2.5s")
    with col2:
        st.metric("Resolution Rate", "85%")
    with col3:
        st.metric("Customer Satisfaction", "4.3/5")

    st.write("### Trend Analysis")
    data = pd.DataFrame({
        "Time": ["00:00", "06:00", "12:00", "18:00", "24:00"],
        "Queries": [50, 200, 150, 180, 100]
    })
    fig, ax = plt.subplots()
    ax.bar(data["Time"], data["Queries"])
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Number of Queries")
    st.pyplot(fig)

    st.write("### Generate Custom Reports")
    report_type = st.selectbox("Select Report Type", ["Unresolved Queries", "Escalated Issues", "Customer Feedback"])
    date_range = st.date_input("Select Date Range", [])
    if st.button("Generate Report"):
        st.success(f"Report generated for {report_type} in the selected date range.")

# Conversation Logs
def conversation_logs(conversation_logs):
    st.subheader("📜 Conversation Logs")
    st.write("### Search and Filter")
    col1, col2, col3 = st.columns(3)
    with col1:
        search_query = st.text_input("Search by Customer Name or Order ID")
    with col2:
        filter_status = st.selectbox("Filter by Resolution Status", ["All", "Resolved", "Unresolved", "Escalated"])
    with col3:
        filter_date = st.date_input("Filter by Date")

    # Filter logs
    if search_query:
        conversation_logs = conversation_logs[
            (conversation_logs["Customer Name"].str.contains(search_query, case=False)) |
            (conversation_logs["Order ID"].str.contains(search_query, case=False))
        ]
    if filter_status != "All":
        conversation_logs = conversation_logs[conversation_logs["Resolution Status"] == filter_status]
    if filter_date:
        conversation_logs = conversation_logs[conversation_logs["Date"] == str(filter_date)]

    st.write("### Chat Transcripts")
    st.dataframe(conversation_logs)

    if st.button("Export Logs"):
        st.success("Logs exported successfully.")

# FAQ Management
def faq_management(faq_data):
    st.subheader("📚 FAQ Management")
    st.write("### FAQ List")
    st.dataframe(faq_data)

    st.write("### Add/Edit FAQs")
    with st.form("faq_form"):
        question = st.text_input("Question")
        answer = st.text_area("Answer")
        category = st.selectbox("Category", ["Shipping", "Returns", "Payments"])
        if st.form_submit_button("Save FAQ"):
            new_faq = {"Question": question, "Answer": answer, "Category": category, "Usage Count": 0}
            # Use pd.concat to append the new FAQ
            faq_data = pd.concat([faq_data, pd.DataFrame([new_faq])], ignore_index=True)
            st.success("FAQ saved successfully.")

# Configuration Section
def chatbot_configuration():
    st.subheader("⚙️ Chatbot Configuration")
    st.write("### Adjust Chatbot Behavior")

    # Example configuration options
    chatbot_prompt = st.text_area(
        "Chatbot Prompt",
        value="The following is a conversation with an e-commerce customer service chatbot. "
              "The chatbot is helpful, polite, and provides accurate responses.\n\n"
              "Context from the knowledge base:\n{context}\n\n"
              "User Query: {query}\n"
              "Chatbot Response:",
        height=200
    )

    chatbot_tone = st.selectbox(
        "Chatbot Tone",
        options=["Friendly", "Professional", "Formal", "Casual"]
    )

    if st.button("Save Configuration"):
        st.session_state.chatbot_prompt = chatbot_prompt
        st.session_state.chatbot_tone = chatbot_tone
        st.success("Chatbot configuration saved successfully.")

# Admin Page
def admin_page():
    st.title("🛠️ System Administrator Dashboard")
    st.write(f"Welcome, {st.session_state.username}!")

    # Top Navigation Bar
    st.write("### Quick Links")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Dashboard"):
            st.session_state.admin_section = "Dashboard"
    with col2:
        if st.button("Configuration"):
            st.session_state.admin_section = "Configuration"
    with col3:
        if st.button("Logs"):
            st.session_state.admin_section = "Logs"
    with col4:
        if st.button("Analytics"):
            st.session_state.admin_section = "Analytics"
    with col5:
        if st.button("FAQ Management"):
            st.session_state.admin_section = "FAQ Management"

    # Sidebar Menu
    st.sidebar.title("Navigation")
    admin_section = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Configuration", "Logs", "Analytics", "FAQ Management"]
    )

    # Load simulated data
    conversation_logs, faq_data = load_admin_data()

    # Main Content Area
    if admin_section == "Dashboard":
        dashboard_overview()
    elif admin_section == "Configuration":
        chatbot_configuration()
    elif admin_section == "Logs":
        conversation_logs(conversation_logs)
    elif admin_section == "Analytics":
        analytics_and_reporting()
    elif admin_section == "FAQ Management":
        faq_management(faq_data)

    # Logout Button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.page = "login"  # Navigate back to Login Page
        st.rerun()

# Main App
def main():
    # Initialize session state for authentication and page navigation
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "role" not in st.session_state:
        st.session_state.role = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "page" not in st.session_state:
        st.session_state.page = "login"  # Default to Login Page

    # Render the appropriate page based on session state
    if st.session_state.page == "login":
        login_signup_page()
    elif st.session_state.page == "chatbot":
        chatbot_page()
    elif st.session_state.page == "admin":
        admin_page()

if __name__ == "__main__":
    main()