# Streamlit UI

Documentation for the interactive Streamlit user interface.

## Overview

The Streamlit UI provides a user-friendly interface for MBTI personality predictions with:

- Text input for prediction
- Real-time API calls
- Interactive radar plot visualization
- Probability scores display

## Features

### Text Input

Large text area for entering writing samples:

```python
text_input = st.text_area(
    "Enter your text",
    height=200,
    placeholder="Write about your thoughts, interests, and personality..."
)
```

**Recommendations:**
- Minimum 50 characters
- Optimal: 200-500 words
- Include varied topics for best results

### Prediction Display

Results shown in three sections:

1. **Predicted Type**: Large heading with MBTI type
2. **Radar Plot**: Visual representation of dimensions
3. **Probability Scores**: Detailed percentages

### Radar Plot

Interactive Plotly radar chart showing:

- 4 axes (E/I, S/N, T/F, J/P)
- Scores from 0-1
- Color-coded areas
- Hover tooltips

## Configuration

### API URL

Set via environment variable:

```bash
# Local development
export API_URL=http://localhost:8000

# Docker
export API_URL=http://api:8000

# Remote
export API_URL=https://api.your-domain.com
```

In `docker-compose.yaml`:

```yaml
ui:
  environment:
    - API_URL=http://api:8000
```

### Page Config

```python
st.set_page_config(
    page_title="MBTI Classifier",
    page_icon="🧠",
    layout="wide",
)
```

## Usage Examples

### Running Locally

```bash
# Start API first
uv run invoke api

# Then start UI
uv run invoke ui

# Access at http://localhost:8501
```

### Running in Docker

```bash
# Start both services
docker compose up -d api ui

# Access at http://localhost:8501
```

### Custom Port

```bash
# Use different port
uv run invoke ui --port 8502

# Or directly
uv run streamlit run src/mbti_classifier/ui.py --server.port 8502
```

## Customization

### Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Layout

Modify `ui.py`:

```python
# Sidebar for additional info
with st.sidebar:
    st.header("About")
    st.write("MBTI personality prediction using DistilBERT")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    # Prediction area
    pass

with col2:
    # Additional info
    pass
```

## Error Handling

The UI handles various errors gracefully:

### API Unreachable

```python
try:
    response = requests.post(...)
except requests.ConnectionError:
    st.error("❌ Cannot connect to API. Make sure it's running on port 8000.")
```

### Invalid Input

```python
if len(text_input) < 10:
    st.warning("⚠️ Please enter more text for accurate prediction.")
```

### API Errors

```python
if response.status_code != 200:
    st.error(f"❌ Prediction failed: {response.json()['detail']}")
```

## Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set `API_URL` in secrets

### Docker

Already configured in `docker-compose.yaml`:

```yaml
ui:
  build:
    context: .
    dockerfile: dockerfiles/ui.dockerfile
  ports:
    - "8501:8501"
  environment:
    - API_URL=http://api:8000
```

## Performance

### Caching

Streamlit caches unchanged data:

```python
@st.cache_data
def get_api_health():
    response = requests.get(f"{API_URL}/health")
    return response.json()
```

### Session State

Preserve user input across reruns:

```python
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Add new prediction
st.session_state.prediction_history.append(result)
```

## Monitoring

### User Analytics

Add analytics to track usage:

```python
import streamlit.components.v1 as components

# Google Analytics
components.html("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
""", height=0)
```

### Error Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    response = requests.post(...)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    st.error("Something went wrong. Please try again.")
```

## Next Steps

- Read [API Reference](api.md) for backend details
- Explore [Docker](../development/docker.md) deployment
- Learn about [Configuration](../configuration/hydra.md)
