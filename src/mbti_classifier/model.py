import logging

import torch
from transformers import DistilBertForSequenceClassification

# configuring the loggs
logger = logging.getLogger(__name__)


### dummy model - testing
def build_model(num_labels: int = 2):
    model_name = "distilbert-base-uncased"
    logger.info(f"Loading architecture: {model_name}")

    # Usamos la clase de Hugging Face que ya tiene la "cabeza" de clasificación
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  ### prueba

    # 1. model
    model = build_model()

    # 2. Cfalse dta to test the model - sanity checkkk
    # we simualate a sentence of 10 words
    dummy_input = torch.randint(0, 1000, (1, 10))

    # 3. a forward pass
    output = model(dummy_input)

    print(f"correctly loaded model: {type(model).__name__}")
    print(f"Output shape (Logits): {output.logits.shape}")
    # Debería salir [1, 2] -> 1 frase, 2 probabilidades (Clase 0 vs Clase 1)
