import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set page config
st.set_page_config(
    page_title="RNN Next-Word Prediction",
    page_icon="🧠",
    layout="wide"
)

# Suppress TensorFlow warnings for cleaner deployment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .highlight-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 RNN Next-Word Prediction Demo</h1>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("🔧 Model Configuration")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.max_len = None
    st.session_state.vocab_size = None
    st.session_state.training_history = None

# Sidebar controls
epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=20, value=8)
embedding_dim = st.sidebar.slider("Embedding Dimension", min_value=16, max_value=128, value=32, step=16)
lstm_units = st.sidebar.slider("LSTM Units", min_value=32, max_value=256, value=64, step=32)
repetitions = st.sidebar.slider("Data Repetitions", min_value=10, max_value=100, value=50, step=10)

# Custom training data option
st.sidebar.subheader("📝 Training Data")
use_custom_data = st.sidebar.checkbox("Use Custom Training Data")

if use_custom_data:
    custom_phrases = st.sidebar.text_area(
        "Enter phrases (one per line):",
        value="""our company provides excellent service
we deliver quality products
the team works hard
our mission is success
we help customers grow
the company values integrity
our products are innovative
we provide great support""",
        height=200
    )
    business_phrases = [phrase.strip() for phrase in custom_phrases.split('\n') if phrase.strip()]
else:
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

# Train model function
@st.cache_data
def prepare_data(phrases, reps):
    """Prepare training data from phrases"""
    corpus = phrases * reps
    
    # Tokenize
    tok = Tokenizer(num_words=1000, oov_token="<OOV>")
    tok.fit_on_texts(corpus)
    seqs = tok.texts_to_sequences(corpus)
    
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
    
    return X, y, tok, max_len

@st.cache_resource
def train_model(X, y, vocab_size, max_len, embedding_dim, lstm_units, epochs):
    """Train the RNN model - cached for better performance"""
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(vocab_size, embedding_dim, mask_zero=True),
        LSTM(lstm_units),
        Dense(vocab_size, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Use a progress bar for training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    class TrainingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f'Training... Epoch {epoch + 1}/{epochs}')
    
    history = model.fit(
        X, y, 
        epochs=epochs, 
        batch_size=64, 
        validation_split=0.2,
        verbose=0,
        callbacks=[TrainingCallback()]
    )
    
    progress_bar.empty()
    status_text.empty()
    
    return model, history

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="section-header">📊 Model Training</h2>', unsafe_allow_html=True)
    
    # Display training data info
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.write(f"**Training Phrases:** {len(business_phrases)}")
    st.write(f"**Total Training Examples:** {len(business_phrases) * repetitions}")
    
    with st.expander("View Training Phrases"):
        for i, phrase in enumerate(business_phrases, 1):
            st.write(f"{i}. {phrase}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Train button
    if st.button("🚀 Train RNN Model", type="primary"):
        with st.spinner("Training RNN model... This may take a moment."):
            # Prepare data
            X, y, tokenizer, max_len = prepare_data(business_phrases, repetitions)
            vocab_size = min(1000, len(tokenizer.word_index) + 1)
            
            # Train model
            model, history = train_model(X, y, vocab_size, max_len, embedding_dim, lstm_units, epochs)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.max_len = max_len
            st.session_state.vocab_size = vocab_size
            st.session_state.training_history = history
            st.session_state.model_trained = True
            
            st.success("✅ Model trained successfully!")

with col2:
    st.markdown('<h2 class="section-header">📈 Training Progress</h2>', unsafe_allow_html=True)
    
    if st.session_state.model_trained and st.session_state.training_history:
        history = st.session_state.training_history
        
        # Create training plots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training Loss', 'Training Accuracy'),
            vertical_spacing=0.15
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(
                y=history.history['loss'],
                name='Training Loss',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        if 'val_loss' in history.history:
            fig.add_trace(
                go.Scatter(
                    y=history.history['val_loss'],
                    name='Validation Loss',
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(
                y=history.history['accuracy'],
                name='Training Accuracy',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        if 'val_accuracy' in history.history:
            fig.add_trace(
                go.Scatter(
                    y=history.history['val_accuracy'],
                    name='Validation Accuracy',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=400, showlegend=True)
        fig.update_xaxes(title_text="Epoch")
        
        st.plotly_chart(fig, use_container_width=True)

# Prediction Section
if st.session_state.model_trained:
    st.markdown('<h2 class="section-header">🔮 Next-Word Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Interactive Prediction")
        
        # Input text
        input_text = st.text_input(
            "Enter text for next-word prediction:",
            value="our company",
            help="Enter a phrase and see what the RNN predicts as the next word"
        )
        
        # Number of predictions to show
        top_k = st.slider("Number of predictions to show", 1, 10, 5)
        
        if input_text:
            # Tokenize and predict
            tokenizer = st.session_state.tokenizer
            model = st.session_state.model
            max_len = st.session_state.max_len
            
            # Prepare input
            seq = tokenizer.texts_to_sequences([input_text.lower()])
            if seq[0]:  # Check if tokenization was successful
                seq = pad_sequences(seq, maxlen=max_len, padding="pre")
                
                # Get predictions
                predictions = model.predict(seq, verbose=0)[0]
                
                # Get top-k predictions
                top_indices = predictions.argsort()[-top_k:][::-1]
                
                # Create index to word mapping
                index_to_word = {i: w for w, i in tokenizer.word_index.items()}
                index_to_word[0] = "<PAD>"  # For padding
                if tokenizer.oov_token:
                    index_to_word[1] = tokenizer.oov_token
                
                # Display predictions
                st.markdown("**Top Predictions:**")
                
                prediction_data = []
                for i, idx in enumerate(top_indices):
                    word = index_to_word.get(idx, f"<{idx}>")
                    prob = predictions[idx]
                    prediction_data.append({
                        'Rank': i + 1,
                        'Word': word,
                        'Probability': f"{prob:.3f}",
                        'Percentage': f"{prob * 100:.1f}%"
                    })
                    
                    # Visual representation
                    st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"**{i+1}. {word}** - {prob:.3f} ({prob*100:.1f}%)")
                    st.progress(prob)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show as dataframe
                df = pd.DataFrame(prediction_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("⚠️ The input text contains words not seen during training!")
    
    with col2:
        st.subheader("Probability Distribution")
        
        if input_text and st.session_state.model_trained:
            # Create probability distribution chart
            tokenizer = st.session_state.tokenizer
            model = st.session_state.model
            max_len = st.session_state.max_len
            
            seq = tokenizer.texts_to_sequences([input_text.lower()])
            if seq[0]:
                seq = pad_sequences(seq, maxlen=max_len, padding="pre")
                predictions = model.predict(seq, verbose=0)[0]
                
                # Get top words for visualization
                top_indices = predictions.argsort()[-10:][::-1]
                index_to_word = {i: w for w, i in tokenizer.word_index.items()}
                index_to_word[0] = "<PAD>"
                if tokenizer.oov_token:
                    index_to_word[1] = tokenizer.oov_token
                
                words = [index_to_word.get(idx, f"<{idx}>") for idx in top_indices]
                probs = [predictions[idx] for idx in top_indices]
                
                # Create bar chart
                fig = px.bar(
                    x=probs,
                    y=words,
                    orientation='h',
                    title=f'Top 10 Predictions for "{input_text}"',
                    labels={'x': 'Probability', 'y': 'Words'},
                    color=probs,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)

    # Preset examples section
    st.markdown('<h3 class="section-header">💡 Try These Examples</h3>', unsafe_allow_html=True)
    
    example_cols = st.columns(4)
    examples = ["our", "our company", "we provide", "the team"]
    
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(f'"{example}"', key=f"example_{i}"):
                st.session_state.example_text = example
                st.rerun()

# Model Architecture Visualization
if st.session_state.model_trained:
    st.markdown('<h2 class="section-header">🏗️ Model Architecture</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Model Components:**
        
        1. **Input Layer**: Receives padded sequences of token IDs
        2. **Embedding Layer**: Converts tokens to dense vectors
        3. **LSTM Layer**: Processes sequences and maintains memory
        4. **Dense Layer**: Outputs probability distribution over vocabulary
        """)
        
        if st.session_state.model:
            model = st.session_state.model
            
            st.markdown("**Model Summary:**")
            
            # Create a simple model summary
            summary_data = []
            for i, layer in enumerate(model.layers):
                summary_data.append({
                    'Layer': layer.name,
                    'Type': layer.__class__.__name__,
                    'Output Shape': str(layer.output_shape),
                    'Parameters': layer.count_params()
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            total_params = model.count_params()
            st.metric("Total Parameters", f"{total_params:,}")
    
    with col2:
        st.markdown("**How RNN Predicts Next Word:**")
        st.markdown("""
        1. **Tokenization**: Convert text to numbers
        2. **Embedding**: Map tokens to dense vectors
        3. **Sequential Processing**: LSTM reads left-to-right
        4. **Context Building**: Hidden state accumulates information
        5. **Prediction**: Final state predicts next word probabilities
        """)
        
        # Show tokenization example
        if st.session_state.tokenizer:
            st.markdown("**Tokenization Example:**")
            example_text = "our company provides"
            tokenizer = st.session_state.tokenizer
            tokens = tokenizer.texts_to_sequences([example_text])[0]
            
            token_df = pd.DataFrame({
                'Word': example_text.split(),
                'Token ID': tokens
            })
            st.dataframe(token_df, use_container_width=True)

else:
    st.info("👆 Train the model first to see predictions and architecture details!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with ❤️ using Streamlit and TensorFlow | 
    Understanding RNNs through Interactive Learning
</div>
""", unsafe_allow_html=True)
