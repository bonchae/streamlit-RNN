import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import plotly.express as px
import os

# Set page config
st.set_page_config(
    page_title="RNN Next-Word Prediction",
    page_icon="🧠",
    layout="wide"
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .example-button {
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 RNN Next-Word Prediction Demo</h1>', unsafe_allow_html=True)

# Load pre-trained model
@st.cache_resource
def load_model():
    """Load and train the RNN model once"""
    # Training data
    business_phrases = [
        "our company provides excellent service",
        "we deliver quality products", 
        "the team works hard",
        "our mission is success",
        "we help customers grow",
        "the company values integrity",
        "our products are innovative",
        "we provide great support"
    ]
    
    # Prepare data
    corpus = business_phrases * 50
    
    # Tokenize
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(corpus)
    seqs = tokenizer.texts_to_sequences(corpus)
    
    # Build training pairs
    X_list, y_list = [], []
    for seq in seqs:
        for i in range(1, len(seq)):
            X_list.append(seq[:i])
            y_list.append(seq[i])
    
    # Pad sequences
    max_len = max(len(s) for s in X_list)
    X = pad_sequences(X_list, maxlen=max_len, padding="pre")
    y = np.array(y_list, dtype=np.int32)
    
    # Create model
    vocab_size = min(1000, len(tokenizer.word_index) + 1)
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(vocab_size, 32, mask_zero=True),
        LSTM(64),
        Dense(vocab_size, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Train model
    model.fit(X, y, epochs=8, batch_size=64, verbose=0)
    
    return model, tokenizer, max_len

def predict_next_words(model, tokenizer, max_len, input_text, top_k=5):
    """Predict next words for given input text"""
    # Tokenize input
    seq = tokenizer.texts_to_sequences([input_text.lower()])
    
    if not seq[0]:  # Empty sequence
        return []
    
    # Pad sequence
    seq = pad_sequences(seq, maxlen=max_len, padding="pre")
    
    # Get predictions
    predictions = model.predict(seq, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = predictions.argsort()[-top_k:][::-1]
    
    # Create word mapping
    index_to_word = {i: w for w, i in tokenizer.word_index.items()}
    index_to_word[0] = "<PAD>"
    if tokenizer.oov_token:
        index_to_word[1] = tokenizer.oov_token
    
    # Build results
    results = []
    for idx in top_indices:
        word = index_to_word.get(idx, f"<{idx}>")
        prob = float(predictions[idx])  # Convert to Python float
        results.append({
            'word': word,
            'probability': prob,
            'percentage': prob * 100
        })
    
    return results

# Load model (this will be cached)
with st.spinner("🔄 Loading RNN model... (This happens once)"):
    model, tokenizer, max_len = load_model()

st.success("✅ Model loaded successfully!")

# Main prediction interface
st.markdown("## 🔮 Interactive Next-Word Prediction")

col1, col2 = st.columns([2, 1])

with col1:
    # Input section
    st.markdown("### Enter your text:")
    input_text = st.text_input(
        "",
        value="our company",
        placeholder="Type your text here...",
        help="Enter a phrase and see what the RNN predicts as the next word"
    )
    
    # Quick examples
    st.markdown("**Quick Examples:**")
    example_cols = st.columns(4)
    examples = ["our", "our company", "we provide", "the team"]
    
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(f'"{example}"', key=f"example_{i}"):
                st.session_state.input_text = example
                st.rerun()
    
    # Use session state if example was clicked
    if 'input_text' in st.session_state:
        input_text = st.session_state.input_text
        # Clear the session state
        del st.session_state.input_text
    
    # Number of predictions
    top_k = st.slider("Number of predictions to show:", 1, 10, 5)

with col2:
    st.markdown("### Model Info:")
    st.info("""
    **Architecture:** Embedding → LSTM → Dense
    
    **Training Data:** Business phrases
    
    **Task:** Predict the most likely next word
    """)

# Predictions section
if input_text.strip():
    st.markdown("---")
    st.markdown(f"### 📊 Predictions for: *\"{input_text}\"*")
    
    # Get predictions
    results = predict_next_words(model, tokenizer, max_len, input_text, top_k)
    
    if results:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Top Predictions:**")
            
            for i, result in enumerate(results):
                word = result['word']
                prob = result['probability']
                percentage = result['percentage']
                
                # Create a styled prediction box
                st.markdown(f"""
                <div class="prediction-box">
                    <strong>{i+1}. {word}</strong><br>
                    Probability: {prob:.3f} ({percentage:.1f}%)
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar (convert to Python float to avoid the error)
                st.progress(float(prob))
        
        with col2:
            st.markdown("**Probability Distribution:**")
            
            # Create probability chart
            words = [r['word'] for r in results]
            probs = [r['probability'] for r in results]
            
            fig = px.bar(
                x=probs,
                y=words,
                orientation='h',
                title=f'Top {top_k} Predictions',
                labels={'x': 'Probability', 'y': 'Words'},
                color=probs,
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                height=400,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.markdown("**Detailed Results:**")
        df = pd.DataFrame([
            {
                'Rank': i + 1,
                'Word': r['word'],
                'Probability': f"{r['probability']:.4f}",
                'Percentage': f"{r['percentage']:.1f}%"
            }
            for i, r in enumerate(results)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
    else:
        st.warning("⚠️ No predictions available. The input might contain unknown words.")

else:
    st.info("👆 Enter some text above to see predictions!")

# How it works section
st.markdown("---")
st.markdown("## 🧠 How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **1. Tokenization**
    
    Text is converted to numbers that the model can understand.
    
    Example: "our company" → [2, 4]
    """)

with col2:
    st.markdown("""
    **2. RNN Processing**
    
    The LSTM processes the sequence left-to-right, building context.
    
    Each word updates the hidden state.
    """)

with col3:
    st.markdown("""
    **3. Prediction**
    
    The final hidden state predicts probabilities for all possible next words.
    
    Higher probability = more likely next word.
    """)

# Sample tokenization
if input_text.strip():
    st.markdown("### 🔤 Tokenization Example")
    
    # Show how the input gets tokenized
    seq = tokenizer.texts_to_sequences([input_text.lower()])[0]
    words = input_text.lower().split()
    
    if seq and len(words) == len(seq):
        token_df = pd.DataFrame({
            'Word': words,
            'Token ID': seq
        })
        st.dataframe(token_df, use_container_width=True, hide_index=True)
    else:
        st.write(f"Input tokens: {seq}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    🎓 Educational RNN Demo | Built with Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)
