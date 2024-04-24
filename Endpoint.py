from flask import Flask, request, jsonify, abort
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
# Load the Sentence Transformer model
model = SentenceTransformer('thenlper/gte-large')
# Precomputed embeddings and DataFrame are initialized as None
category_embeddings = None
families_df = None
def load_data_and_embeddings():
   global families_df, category_embeddings
   try:
       # Load category data
       families_df = pd.read_excel('families.xlsx')
       # Precompute embeddings for all categories
       category_embeddings = model.encode(families_df['English Description'].tolist())
   except Exception as e:
       print("Failed to load families.xlsx or compute embeddings:", e)
       exit(1)
@app.before_first_request
def initialize():
   """ Initialize data and embeddings before the first request is processed. """
   load_data_and_embeddings()
def find_in_sources(part_number, source_data):
   """Search for the part number in predefined source data and return the corresponding family if found."""
   for source, df in source_data.items():
       match = df[df['Part Number'] == part_number]
       if not match.empty:
           return match.iloc[0]['Family'], source
   return None, None
def batch_classify(descriptions):
   """Classify descriptions in batches using zero-shot classification based on cosine similarity."""
   embeddings = model.encode(descriptions)
   similarities = cosine_similarity(embeddings, category_embeddings)
   indices = np.argmax(similarities, axis=1)
   return families_df['Family'].values[indices]
def process_descriptions(data, source_data):
   """Process each description, search in sources by part number, validate, and classify if necessary."""
   needs_zero_shot = []
   indices = []
   for idx, row in data.iterrows():
       part_number = row['Part Number']
       family, source = find_in_sources(part_number, source_data)
       if family:
           data.at[idx, 'Proposed Family'] = family
           data.at[idx, 'Found in'] = source
       else:
           description = row['Material Description']
           if pd.isna(description) or description.strip().lower() in ['unknown', 'missing', 'blank', ''] or description.strip().isdigit() or len(description.strip()) < 4:
               data.at[idx, 'Proposed Family'] = 'Unable to identify'
               data.at[idx, 'Found in'] = 'Unable to identify'
           else:
               needs_zero_shot.append(description)
               indices.append(idx)
   if needs_zero_shot:
       categories = batch_classify(needs_zero_shot)
       for idx, category in zip(indices, categories):
           data.at[idx, 'Proposed Family'] = category
           data.at[idx, 'Found in'] = 'AI Classification'
   return data
@app.route('/process_file', methods=['POST'])
def process_file():
   if 'file' not in request.files:
       return "No file part", 400
   file = request.files['file']
   if file.filename == '':
       return "No selected file", 400
   data = pd.read_excel(file)
   # Load external source data here as needed
   sources = ['CSM', 'iNEGO', 'PRISM', 'CAP Tool']
   source_data = {source: pd.read_csv(f'{source}_data.csv', encoding='ISO-8859-1') for source in sources}
   processed_data = process_descriptions(data, source_data)
   response = processed_data.to_json()
   return response
@app.route('/classify_description', methods=['POST'])
def classify_description():
   data = request.get_json()
   descriptions = data.get('descriptions', [])
   if not descriptions:
       return jsonify({'error': 'No descriptions provided'}), 400
   categories = batch_classify(descriptions)
   return jsonify({'categories': categories.tolist()})
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')
