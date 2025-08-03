import os
import pandas as pd
import torch
import logging
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import easyocr
from transformers import RobertaModel, RobertaTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl import load_workbook
import concurrent.futures



from transformers import logging as transformers_logging
# Paths and filenames
input_file = "TrainingSet_Modified.xlsx"
output_file = "features_output.xlsx"
images_dir = "images"
html_dir = "Alloutputs"

device = "cuda" if torch.cuda.is_available() else "cpu"

# RoBERTa setup
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
if hasattr(roberta_model, 'lstm'):
    roberta_model.lstm.flatten_parameters()

recognizer_model = ocr_reader.recognizer

# Check if DataParallel exists and unwrap it
if hasattr(recognizer_model, 'module'):
    recognizer_model = recognizer_model.module

# Flatten parameters if 'rnn' layer is found
if hasattr(recognizer_model, 'model') and hasattr(recognizer_model.model, 'rnn'):
    recognizer_model.model.rnn.flatten_parameters()

# Optionally clean GPU memory if using CUDA to optimize memory usage
if torch.cuda.is_available():
    print("Using GPU")
    torch.cuda.empty_cache()



class CustomFeatureClassifier(torch.nn.Module):
    def init(self, text_feature_dim=768, num_labels=2):
        super(CustomFeatureClassifier, self).init()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.feature_projection = torch.nn.Linear(text_feature_dim, self.roberta.config.hidden_size)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(0.2)  # Adjust dropout
        self.num_labels = num_labels

    def forward(self, features, labels=None):
        projected_features = self.feature_projection(features).unsqueeze(1)
        outputs = self.roberta(inputs_embeds=projected_features)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
def extract_text_features(text):
    if not text:
        return np.zeros(768).tolist()
    try:
        inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = roberta_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten().tolist()
    except:
        return np.zeros(768).tolist()

# Load dataset
df = pd.read_excel(input_file)
required_columns = ['URL', 'label', 'status', 'html_file', 'image_file']
df = df.dropna(subset=['html_file'])

features_list = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing URLs"):
    url, label, status, html_file, image_file = row[required_columns]

    html_path = os.path.join(html_dir, str(html_file))
    if not os.path.isfile(html_path):
        continue

    text_data = ""

    # Extract text from image if folder exists
    if pd.notna(image_file):
        image_folder_path = os.path.join(images_dir, str(image_file))
        if os.path.isdir(image_folder_path):
            try:
                for file in os.listdir(image_folder_path):
                    if file.lower().endswith(".png"):
                        image_path = os.path.join(image_folder_path, file)
                        image = Image.open(image_path).convert("RGB")
                        ocr_results = ocr_reader.readtext(np.array(image), detail=0)
                        text_data += " ".join(ocr_results) + " "
            except:
                pass

    # Read HTML file text
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_data += " " + f.read()
    except:
        continue

    features = extract_text_features(text_data)
    features_list.append({
        'URL': url,
        'label': 1 if label.lower() == 'phishing' else 0,
        'status': status,
        'features': features
    })

features_df = pd.DataFrame(features_list)
features_df.to_excel(output_file, index=False)

df = pd.read_excel(output_file)

# Shuffle the rows
df_shuffled = df.sample(frac=1, random_state=random.randint(0, 34000)).reset_index(drop=True)

# Overwrite the original file
df_shuffled.to_excel(output_file, index=False)

import ast
# Split the dataset
# Load the Excel file
Model_features_df = pd.read_excel("features_output.xlsx")

# Convert 'features' from string to actual list
Model_features_df['features'] = Model_features_df['features'].apply(ast.literal_eval)

# Now convert to numpy arrays
X = np.array(Model_features_df['features'].tolist(), dtype=np.float32)
y = Model_features_df['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


# Dataset class
class FeatureDataset(torch.utils.data.Dataset):
    def init(self, features, labels):
        self.features = features
        self.labels = labels

    def getitem(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def len(self):
        return len(self.labels)

train_dataset = FeatureDataset(X_train, y_train)
test_dataset = FeatureDataset(X_test, y_test)

# Train the model
model = CustomFeatureClassifier().to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=425,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    logging_dir="./logs",
    eval_strategy='epoch',
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
    disable_tqdm=False,
    logging_strategy="epoch"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
metrics = trainer.evaluate()
print(f"Test Accuracy: {metrics['eval_accuracy']:.4f}")
