import streamlit as st
import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import ollama
import tensorflow as tf
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
import psycopg2
from psycopg2 import sql

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Ollama client
client = ollama.Client()

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
def fetch_customer_by_email(email):
    """Fetch a customer by their email from the customer table."""
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM customer WHERE email = %s;", (email,))
            customer = cur.fetchone()
            cur.close()
            conn.close()
            return customer
        except Exception as e:
            print(f"Error fetching customer by email: {e}")
            return None
        
# Function to fetch a user by their ID
def fetch_customer_by_id(cust_id):
    """Fetch a customer by their ID from the customer table."""
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM customer WHERE cust_id = %s;", (cust_id,))
            customer = cur.fetchone()
            cur.close()
            conn.close()
            return customer
        except Exception as e:
            print(f"Error fetching customer by ID: {e}")
            return None

# Function to fetch all users
def fetch_all_customers():
    """Fetch all customer names from the customer table."""
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM customer;")
            customers = cur.fetchall()
            cur.close()
            conn.close()
            return customers
        except Exception as e:
            print(f"Error fetching customers: {e}")
            return []

# Function to fetch a product by its ID
def fetch_product_by_id(product_id):
    """Fetch a product by its ID from the products table."""
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM products WHERE product_id = %s;", (product_id,))
            product = cur.fetchone()
            cur.close()
            conn.close()
            return product
        except Exception as e:
            print(f"Error fetching product by ID: {e}")
            return None
# Function to fetch all products
def fetch_all_products():
    """Fetch all product names from the products table."""
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM products;")
            products = cur.fetchall()
            cur.close()
            conn.close()
            return products
        except Exception as e:
            print(f"Error fetching products: {e}")
            return []

# Function to fetch an order by its ID
def fetch_order_by_id(order_id):
    """Fetch an order by its ID from the orders table."""
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM orders WHERE order_id = %s;", (order_id,))
            order = cur.fetchone()
            cur.close()
            conn.close()
            return order
        except Exception as e:
            print(f"Error fetching order by ID: {e}")
            return None

# Function to fetch all orders
def fetch_all_orders():
    """Fetch all order IDs from the orders table."""
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT order_id FROM orders;")
            orders = cur.fetchall()
            cur.close()
            conn.close()
            return orders
        except Exception as e:
            print(f"Error fetching orders: {e}")
            return []
# Load knowledge base and embeddings
@st.cache_data
def load_data():
    embeddings = np.load("question_embeddings.npy")  # Update with your file path
    knowledge_base = pd.read_csv("knowledge_base_text.csv")  # Update with your file path
    return embeddings, knowledge_base

# Initialize FAISS index
@st.cache_resource
def initialize_faiss(embeddings):
    dimension = embeddings.shape[1]  # Embedding size
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Load the model and tokenizer for query embedding using transformers
@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # You can replace this with any Hugging Face model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Load Llama 3.2 model
@st.cache_resource
def load_ollama():
    model = "llama3.2"
    return model

# Load sentiment analysis model and tokenizer
@st.cache_resource
def load_sentiment_model():
    # Load the pre-trained sentiment analysis model
    sentiment_model = load_keras_model("C:/Users/jrosh/DBMS and ANN Project/sentiment_model.h5")  # Update with your file path
    
    # Load the tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    return sentiment_model, tokenizer

# Preprocess text for sentiment analysis
def preprocess_text(sen):
    """Preprocess the text data."""
    sen = re.sub('<.*?>', ' ', sen)  # Remove HTML tags
    tokens = word_tokenize(sen)  # Tokenize words
    tokens = [w.lower() for w in tokens]  # Lowercase
    table = str.maketrans('', '', string.punctuation)  # Remove punctuation
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]  # Remove non-alphabet characters
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]  # Remove stopwords
    words = [w for w in words if len(w) > 2]  # Ignore words less than 2 characters
    return " ".join(words)

# Predict sentiment of a query
def predict_sentiment(query, model, tokenizer, max_length=100):
    """Predict sentiment for a given text."""
    filtered = preprocess_text(query)
    tokenize_words = tokenizer.texts_to_sequences([filtered])
    tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')
    result = model.predict(tokenize_words)
    return 'positive' if result >= 0.5 else 'negative'

# Generate embeddings using transformers
def generate_embeddings(text, tokenizer, model):
    """Generate embeddings for a given text using a transformer model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding as the sentence embedding
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embeddings

# Refine response with Llama 3.2 (for general queries)
def refine_with_ollama(query, retrieved_answers, ollama_model, conversation_history):
    # Format the context from retrieved answers and conversation history
    context = "\n".join([f"Q: {qa[0]}\nA: {qa[1]}" for qa in retrieved_answers])
    history = "\n".join([f"User: {conv[0]}\nChatbot: {conv[1]}" for conv in conversation_history])
    
    # Construct the Llama prompt
    prompt = (
        f"You are a ecommerce customer service chatbot. "
        f"Your tone is supposed to be helpful, polite, and provide accurate responses.\n\n"
        f"Conversation History:\n{history}\n\n"
        f"Context from the knowledge base:\n{context}\n\n"
        f"User Query: {query}\n"
        f"Chatbot Response:"
    )
    
    # Generate response using Ollama
    response = client.generate(model=ollama_model, prompt=prompt)
    response_text = response.get("response", "No response available.")
    return response_text

# Refine response with Llama 3.2 (for complaints mode)
def refine_complaint_response(query, retrieved_answers, ollama_model, sentiment, conversation_history):
    # Format the context from retrieved answers and conversation history
    context = "\n".join([f"Q: {qa[0]}\nA: {qa[1]}" for qa in retrieved_answers])
    history = "\n".join([f"User: {conv[0]}\nChatbot: {conv[1]}" for conv in conversation_history])
    
    # Construct the Llama prompt with sentiment
    if sentiment == 'negative':
        prompt = (
            f"The user is frustrated or angry. The chatbot should respond with empathy and escalate the issue if necessary.\n\n"
            f"Conversation History:\n{history}\n\n"
            f"Context from the knowledge base:\n{context}\n\n"
            f"User Query: {query}\n"
            f"Chatbot Response:"
        )
    else:
        prompt = (
            f"The user has a complaint but is not overly frustrated. The chatbot should provide helpful and polite responses.\n\n"
            f"Conversation History:\n{history}\n\n"
            f"Context from the knowledge base:\n{context}\n\n"
            f"User Query: {query}\n"
            f"Chatbot Response:"
        )
    
    # Generate response using Ollama
    response = client.generate(model=ollama_model, prompt=prompt)
    response_text = response.get("response", "No response available.")
    return response_text

# Retrieve answers and generate a comprehensive response
def retrieve_answer(query, index, knowledge_base, tokenizer, model, ollama_model, sentiment_model=None, tokenizer_sentiment=None, top_k=3, complaints_mode=False):
    # Check if the query is related to fetching user data
    if "customer details" in query.lower() or "customer information" in query.lower():
        # Extract customer ID from the query (e.g., "Tell me about customer 123")
        cust_id = re.search(r'\d+', query)
        if cust_id:
            cust_id = int(cust_id.group())
            customer = fetch_customer_by_id(cust_id)
            if customer:
                return f"Customer details: ID={customer[0]}, Name={customer[1]}, Email={customer[2]}", []
            else:
                return "No customer found with that ID.", []
        else:
            return "Please provide a valid customer ID.", []
    elif "list all customers" in query.lower():
        customers = fetch_all_customers()
        if customers:
            response = "Here are all the customers:\n"
            for customer in customers:
                response += f"Name: {customer[0]}\n"
            return response, []
        else:
            return "No customers found.", []
    elif "product details" in query.lower() or "product information" in query.lower():
        # Extract product ID from the query (e.g., "Tell me about product 123")
        product_id = re.search(r'\d+', query)
        if product_id:
            product_id = int(product_id.group())
            product = fetch_product_by_id(product_id)
            if product:
                return f"Product details: ID={product[0]}, Name={product[1]}, Price={product[2]}", []
            else:
                return "No product found with that ID.", []
        else:
            return "Please provide a valid product ID.", []
    elif "list all products" in query.lower():
        products = fetch_all_products()
        if products:
            response = "Here are all the products:\n"
            for product in products:
                response += f"Name: {product[0]}\n"
            return response, []
        else:
            return "No products found.", []
    elif "order details" in query.lower() or "order information" in query.lower():
        # Extract order ID from the query (e.g., "Tell me about order 123")
        order_id = re.search(r'\d+', query)
        if order_id:
            order_id = int(order_id.group())
            order = fetch_order_by_id(order_id)
            if order:
                return f"Order details: ID={order[0]}, Status={order[1]}, Total Price={order[3]}", []
            else:
                return "No order found with that ID.", []
        else:
            return "Please provide a valid order ID.", []
    elif "list all orders" in query.lower():
        orders = fetch_all_orders()
        if orders:
            response = "Here are all the orders:\n"
            for order in orders:
                response += f"Order ID: {order[0]}\n"
            return response, []
        else:
            return "No orders found.", []

    # Generate embeddings for the query
    query_embedding = generate_embeddings(query, tokenizer, model).reshape(1, -1)
    
    # Retrieve top-k matches from FAISS
    distances, indices = index.search(query_embedding, top_k)
    
    # Gather retrieved answers
    retrieved_answers = []
    for idx in indices[0]:
        question = knowledge_base.iloc[idx]['Question']
        answer = knowledge_base.iloc[idx]['Answer']
        retrieved_answers.append((question, answer))
    
    # Synthesize a comprehensive response
    conversation_history = st.session_state.conversation
    if complaints_mode and sentiment_model and tokenizer_sentiment:
        sentiment = predict_sentiment(query, sentiment_model, tokenizer_sentiment)
        refined_response = refine_complaint_response(query, retrieved_answers, ollama_model, sentiment, conversation_history)
    else:
        refined_response = refine_with_ollama(query, retrieved_answers, ollama_model, conversation_history)
    
    return refined_response, retrieved_answers

# Enhanced UI Design for Results
def display_results(results):
    for i, (matched_question, answer) in enumerate(results, 1):
        st.markdown(f"""
        <div style="background-color:#f9f9f9; padding:15px; margin-bottom:15px; border-radius:10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
            <h4 style="color:#333;">Match {i}</h4>
            <p><strong>Q:</strong> {matched_question}</p>
            <p><strong>A:</strong> {answer}</p>
        </div>
        """, unsafe_allow_html=True)

# Chatbot Page
def chatbot_page():
    # Initialize session state for username if it doesn't exist
    if "username" not in st.session_state:
        st.session_state.username = "User"  # Default value or fetch from login

    # Initialize session state for conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Initialize session state for complaints mode
    if "complaints_mode" not in st.session_state:
        st.session_state.complaints_mode = False

    # Fetch customer details
    customer_details = fetch_customer_by_email(st.session_state.username)
    if customer_details:
        customer_id, customer_name, customer_email, customer_phone, customer_password, customer_role = customer_details
    else:
        customer_name = "User"

    # Sidebar for Navigation
    st.sidebar.image("chatbot_logo.png", use_container_width=True)
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
        - **Ask a Question**: Get customer service answers.
        - **About**: Learn about the chatbot.
        - **Help**: Troubleshooting tips.
    """)

    # Main App Title and Description
    st.title("📞 Customer Service Chatbot with Llama 3.2")
    st.write(f"Welcome, {customer_name}!")  # Display customer name
    st.write("This is the chatbot page. You can interact with the chatbot here.")

    # Load chatbot data and models
    embeddings, knowledge_base = load_data()
    index = initialize_faiss(embeddings)
    tokenizer, model = load_model()
    ollama_model = load_ollama()
    sentiment_model, tokenizer_sentiment = load_sentiment_model()

    # Display previous 3 inputs and responses
    if "conversation" in st.session_state and st.session_state.conversation:
        st.subheader("Previous Conversations")
        for i, (user_query, chatbot_response) in enumerate(st.session_state.conversation[-3:], 1):
            st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:10px; margin-bottom:10px; border-radius:5px;">
                <p><strong>User {i}:</strong> {user_query}</p>
                <p><strong>Chatbot {i}:</strong> {chatbot_response}</p>
            </div>
            """, unsafe_allow_html=True)

    # Search Bar and Submit Button
    if st.session_state.complaints_mode:
        user_query = st.text_area(
            "🔍 Enter your complaint:", 
            placeholder="e.g., Please start your query with 'I have a complaint...'",  
            height=100,  # Adjust height as needed
            key="user_input"
        )
    else:
        user_query = st.text_area(
            "🔍 Enter your query:", 
            placeholder="e.g., How can I track my order?", 
            height=100,  # Adjust height as needed
            key="user_input"
        )

    # Submit Button
    if st.button("Submit", type="primary"):
        if user_query:
            with st.spinner("🔎 Searching for the best match..."):
                refined_response, retrieved_answers = retrieve_answer(
                    user_query, index, knowledge_base, tokenizer, model, ollama_model, 
                    sentiment_model, tokenizer_sentiment, top_k=3, complaints_mode=st.session_state.complaints_mode
                )
            
            # Add current query and response to conversation history
            st.session_state.conversation.append((user_query, refined_response))
            st.markdown("### Chatbot Response")
            st.write(refined_response)

            # Display retrieved matches aligned to the left
            st.markdown("### Retrieved Matches")
            display_results(retrieved_answers)
        else:
            st.warning("Please enter a query before submitting.")
    
    # Complaints Button
    if st.button("Complaints", key="complaints_button"):
        # Toggle complaints mode
        st.session_state.complaints_mode = not st.session_state.complaints_mode
        st.rerun()  # Refresh the app to update the search bar

    # Logout Button
    if st.button("Logout", key="logout_button"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.page = "login"  # Navigate back to Login Page
        st.rerun()

    # Footer Section
    st.write("---")
    st.markdown("""
        <footer style="text-align:center; font-size:small; color:gray;">
            © 2025 Customer Service Chatbot | Powered by Llama 3.2
        </footer>
    """, unsafe_allow_html=True)

# Run the chatbot page
if __name__ == "__main__":
    chatbot_page()