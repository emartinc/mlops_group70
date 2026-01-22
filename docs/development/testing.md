# Testing Guide

This page describes the project's testing strategy, how to execute tests, and how to add new ones.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures (pytest)
├── unit/
│   ├── __init__.py
│   ├── test_data_module.py  # DataModule tests
│   ├── test_model.py        # Model tests
│   └── test_api.py          # API tests
└── integration/
    ├── __init__.py
    └── test_pipeline.py     # End-to-end tests
```

## Testing Commands

### Run All Tests

```bash
# With uv
uv run pytest tests/

# Without uv (if pytest is installed globally)
pytest tests/
```

### Run Tests by Category

```bash
# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# Specific file
uv run pytest tests/unit/test_model.py

# Specific test
uv run pytest tests/unit/test_model.py::test_model_forward_pass
```

### Useful Options

```bash
# Verbose output (shows all tests)
uv run pytest tests/ -v

# Show print statements
uv run pytest tests/ -s

# Stop on first failure
uv run pytest tests/ -x

# Stop after N failures
uv run pytest tests/ --maxfail=3

# Run tests in parallel (requires pytest-xdist)
uv run pytest tests/ -n auto

# Show coverage
uv run pytest tests/ --cov=src/mbti_classifier --cov-report=html
```

### Markers (Tags)

```bash
# Run fast tests only
uv run pytest tests/ -m "not slow"

# Run slow tests only
uv run pytest tests/ -m slow

# Run GPU tests (if available)
uv run pytest tests/ -m gpu
```

## Fixtures

Fixtures in `conftest.py` are available for all tests:

### Example: `conftest.py`

```python
import pytest
import torch
from transformers import DistilBertTokenizer
from src.mbti_classifier.data import MBTIDataModule
from src.mbti_classifier.model import MBTIModel

@pytest.fixture
def tokenizer():
    """DistilBERT Tokenizer."""
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        'posts': ['This is a test post', 'Another test'],
        'E_I': [1, 0],
        'S_N': [0, 1],
        'T_F': [1, 1],
        'J_P': [0, 1]
    }

@pytest.fixture
def model():
    """Model for testing."""
    return MBTIModel(
        model_name='distilbert-base-uncased',
        learning_rate=2e-5,
        dropout_rate=0.1
    )

@pytest.fixture
def datamodule():
    """DataModule for testing."""
    return MBTIDataModule(
        raw_data_path='data/raw/mbti_1.csv',
        processed_data_path='data/processed/processed_mbti.csv',
        batch_size=2,
        max_length=128,
        num_workers=0
    )
```

### Usage in Tests

```python
def test_model_output_shape(model, tokenizer):
    """Test that the model output has the correct shape."""
    # model and tokenizer come from fixtures
    
    # Prepare input
    text = "This is a test"
    encoded = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Forward pass
    output = model(encoded['input_ids'], encoded['attention_mask'])
    
    # Verify shape
    assert output.shape == (1, 4), f"Expected shape (1, 4), got {output.shape}"
```

## Unit Tests

### Data Module Tests: `test_data_module.py`

```python
import pytest
import pandas as pd
from src.mbti_classifier.data import MBTIDataset, MBTIDataModule

class TestMBTIDataset:
    def test_dataset_length(self, sample_data, tokenizer):
        """Test that the dataset has the correct length."""
        df = pd.DataFrame(sample_data)
        dataset = MBTIDataset(df, tokenizer, max_length=128)
        
        assert len(dataset) == len(df)
    
    def test_dataset_item_structure(self, sample_data, tokenizer):
        """Test that each item has the correct keys."""
        df = pd.DataFrame(sample_data)
        dataset = MBTIDataset(df, tokenizer, max_length=128)
        
        item = dataset[0]
        
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
    
    def test_dataset_item_shapes(self, sample_data, tokenizer):
        """Test that tensors have the correct shapes."""
        df = pd.DataFrame(sample_data)
        dataset = MBTIDataset(df, tokenizer, max_length=128)
        
        item = dataset[0]
        
        assert item['input_ids'].shape == (128,)
        assert item['attention_mask'].shape == (128,)
        assert item['labels'].shape == (4,)

class TestMBTIDataModule:
    def test_datamodule_setup(self, datamodule):
        """Test that the datamodule initializes correctly."""
        datamodule.prepare_data()
        datamodule.setup('fit')
        
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
    
    def test_dataloader_batch_size(self, datamodule):
        """Test that dataloaders have the correct batch size."""
        datamodule.prepare_data()
        datamodule.setup('fit')
        
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        assert batch['input_ids'].shape[0] == datamodule.batch_size
```

### Model Tests: `test_model.py`

```python
import pytest
import torch
from src.mbti_classifier.model import MBTIModel

class TestMBTIModel:
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        model = MBTIModel(
            model_name='distilbert-base-uncased',
            learning_rate=2e-5,
            dropout_rate=0.1
        )
        
        assert model is not None
        assert len(model.classifiers) == 4
    
    def test_forward_pass(self, model):
        """Test that the forward pass works."""
        # Dummy input
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)
        
        # Forward
        output = model(input_ids, attention_mask)
        
        # Verify shape
        assert output.shape == (2, 4)
        
        # Verify range (after sigmoid, must be [0, 1])
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_training_step(self, model):
        """Test that the training step works."""
        # Dummy batch
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 128)),
            'attention_mask': torch.ones(2, 128),
            'labels': torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float)
        }
        
        # Training step
        loss = model.training_step(batch, 0)
        
        # Verify that loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
    
    @pytest.mark.slow
    def test_full_training_epoch(self, model, datamodule):
        """Test a full training epoch."""
        import lightning.pytorch as pl
        
        datamodule.prepare_data()
        datamodule.setup('fit')
        
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=10,
            limit_val_batches=5,
            logger=False,
            enable_checkpointing=False
        )
        
        trainer.fit(model, datamodule)
        
        # Verify that it trained without errors
        assert trainer.current_epoch == 1
```

### API Tests: `test_api.py`

```python
import pytest
from fastapi.testclient import TestClient
from src.mbti_classifier.api import app

@pytest.fixture
def client():
    """Testing client for the API."""
    return TestClient(app)

class TestAPI:
    def test_health_endpoint(self, client):
        """Test the health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_predict_endpoint(self, client):
        """Test the prediction endpoint."""
        payload = {
            "text": "I love solving complex problems and analyzing data."
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "mbti_type" in data
        assert len(data["mbti_type"]) == 4
        assert all(c in "EISNTFJP" for c in data["mbti_type"])
    
    def test_predict_batch_endpoint(self, client):
        """Test the batch prediction endpoint."""
        payload = {
            "texts": [
                "I love meeting new people!",
                "I prefer staying home and reading."
            ]
        }
        
        response = client.post("/predict_batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) == 2
    
    def test_predict_invalid_input(self, client):
        """Test that the API rejects invalid inputs."""
        payload = {"text": ""}  # Empty text
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 422  # Validation error
```

## Integration Tests

### Full Pipeline Test: `test_pipeline.py`

```python
import pytest
import os
from pathlib import Path
from src.mbti_classifier.data import download_mbti_data, preprocess_mbti_data
from src.mbti_classifier.model import MBTIClassifier

@pytest.mark.slow
class TestFullPipeline:
    def test_data_download_and_preprocessing(self, tmp_path):
        """Test data download and preprocessing."""
        raw_path = tmp_path / "raw" / "mbti_1.csv"
        processed_path = tmp_path / "processed" / "processed_mbti.csv"
        
        # Download
        download_mbti_data(str(raw_path))
        assert raw_path.exists()
        
        # Preprocess
        preprocess_mbti_data(str(raw_path), str(processed_path))
        assert processed_path.exists()
    
    @pytest.mark.gpu
    def test_training_pipeline(self, tmp_path):
        """Test full training pipeline."""
        import lightning.pytorch as pl
        from src.mbti_classifier.data import MBTIDataModule
        from src.mbti_classifier.model import MBTIModel
        
        # Configure paths
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        # Initialize components
        datamodule = MBTIDataModule(
            raw_data_path='data/raw/mbti_1.csv',
            processed_data_path='data/processed/processed_mbti.csv',
            batch_size=8,
            max_length=128,
            num_workers=0
        )
        
        model = MBTIModel(
            model_name='distilbert-base-uncased',
            learning_rate=2e-5
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=10,
            limit_val_batches=5,
            default_root_dir=str(tmp_path),
            logger=False,
            enable_checkpointing=True
        )
        
        # Train
        trainer.fit(model, datamodule)
        
        # Verify that checkpoint was saved
        checkpoints = list(Path(tmp_path).glob("**/*.ckpt"))
        assert len(checkpoints) > 0
    
    def test_inference_pipeline(self):
        """Test inference pipeline."""
        # Load pre-trained model
        classifier = MBTIClassifier.load_from_checkpoint('models/best.ckpt')
        
        # Predict
        text = "I love analyzing complex problems and finding elegant solutions."
        mbti_type = classifier.predict(text)
        
        # Verify
        assert isinstance(mbti_type, str)
        assert len(mbti_type) == 4
        assert all(c in "EISNTFJP" for c in mbti_type)
```

## Custom Markers

### Define Markers: `pytest.ini`

```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests that require GPU
    integration: marks integration tests
    unit: marks unit tests
```

### Usage in Tests

```python
import pytest

@pytest.mark.slow
def test_long_running_operation():
    """This test takes a long time."""
    # ...

@pytest.mark.gpu
def test_gpu_training():
    """This test requires GPU."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    # ...

@pytest.mark.integration
def test_api_integration():
    """Integration test."""
    # ...
```

### Run with Markers

```bash
# Fast tests only (exclude slow)
uv run pytest -m "not slow"

# GPU tests only
uv run pytest -m gpu

# Integration tests only
uv run pytest -m integration

# Unit + fast
uv run pytest -m "unit and not slow"
```

## Coverage

### Generate Coverage Report

```bash
# Coverage with HTML report
uv run pytest tests/ --cov=src/mbti_classifier --cov-report=html

# Open report
open htmlcov/index.html  # macOS
# Or navigate to htmlcov/index.html in your browser
```

### Terminal Coverage

```bash
uv run pytest tests/ --cov=src/mbti_classifier --cov-report=term-missing
```

**Output:**

```
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/mbti_classifier/data.py    150     10    93%   45-47, 89
src/mbti_classifier/model.py   120      5    96%   67-68
src/mbti_classifier/api.py      80      8    90%   112-115
-----------------------------------------------------------
TOTAL                          350     23    93%
```

### Minimum Coverage

```bash
# Fail if coverage < 80%
uv run pytest tests/ --cov=src/mbti_classifier --cov-fail-under=80
```

## Best Practices

### 1. Use Fixtures for Common Setup

```python
# ✅ Good
@pytest.fixture
def model():
    return MBTIModel(model_name='distilbert-base-uncased')

def test_forward(model):
    # model is already initialized
    output = model(...)

# ❌ Bad
def test_forward():
    model = MBTIModel(model_name='distilbert-base-uncased')  # Repetitive
    output = model(...)
```

### 2. Independent Tests

```python
# ✅ Good: each test is independent
def test_add_user():
    db = Database()
    db.add_user('Alice')
    assert db.count() == 1

def test_remove_user():
    db = Database()
    db.add_user('Bob')
    db.remove_user('Bob')
    assert db.count() == 0

# ❌ Bad: tests depend on order
db = Database()

def test_add_user():
    db.add_user('Alice')
    assert db.count() == 1

def test_remove_user():
    db.remove_user('Alice')  # Assumes test_add_user already ran
    assert db.count() == 0
```

### 3. Use Parametrize for Multiple Cases

```python
# ✅ Good: parametrize
@pytest.mark.parametrize("text,expected", [
    ("I love parties!", "E"),
    ("I prefer staying home", "I"),
    ("Big gatherings are fun", "E")
])
def test_extraversion(text, expected):
    result = classify_ei(text)
    assert result[0] == expected

# ❌ Bad: repetitive tests
def test_extraversion_case1():
    result = classify_ei("I love parties!")
    assert result[0] == "E"

def test_extraversion_case2():
    result = classify_ei("I prefer staying home")
    assert result[0] == "I"
```

### 4. Descriptive Names

```python
# ✅ Good
def test_model_returns_four_probabilities():
    ...

def test_api_rejects_empty_text():
    ...

# ❌ Bad
def test_model():
    ...

def test_api():
    ...
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install uv
      run: curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    
    - name: Install dependencies
      run: uv sync
    
    - name: Run tests
      run: uv run pytest tests/ --cov=src/mbti_classifier --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Troubleshooting

### Problem: Tests are not discovered

**Solution:**
- Verify that files start with `test_`
- Verify that classes start with `Test`
- Verify that `__init__.py` exists in the tests directory

### Problem: Fixture not found

**Solution:**
- Verify that fixture is in `conftest.py`
- Verify fixture scope

### Problem: Slow tests

**Solution:**
```bash
# Run in parallel
uv run pytest tests/ -n auto

# Exclude slow tests
uv run pytest tests/ -m "not slow"
```

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/mark.html)
- [Coverage.py](https://coverage.readthedocs.io/)