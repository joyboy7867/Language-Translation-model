from transformers import MarianMTModel, MarianTokenizer
import torch 
import os
from flask import Flask, request, jsonify
app = Flask(__name__)
def translate_text(text, target_language,reverse=False):
    # Map language codes to model names
    model_names = {
       'fr': ['Helsinki-NLP/opus-mt-en-fr', 'Helsinki-NLP/opus-mt-fr-en'],  # English <-> French
        'de': ['Helsinki-NLP/opus-mt-en-de', 'Helsinki-NLP/opus-mt-de-en'],  # English <-> German
        'es': ['Helsinki-NLP/opus-mt-en-es', 'Helsinki-NLP/opus-mt-es-en'],  # English <-> Spanish
        'it': ['Helsinki-NLP/opus-mt-en-it', 'Helsinki-NLP/opus-mt-it-en'],  # English <-> Italian
        'pt': ['Helsinki-NLP/opus-mt-en-pt', 'Helsinki-NLP/opus-mt-pt-en'],  # English <-> Portuguese
        'zh': ['Helsinki-NLP/opus-mt-en-zh', 'Helsinki-NLP/opus-mt-zh-en'],   # English to Chinese
    }
    models= model_names.get(target_language)
    if not models:
        return "Language not supported."
    model_name = models[1] if reverse else models[0]    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    
    # Decode the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text
@app.route('/translate', methods=['POST'])
def translate():
    # Parse request JSON
    data = request.get_json()
    
    text = data.get('text')  # Text to translate
    target_language = data.get('target_language')  # Target language code
    reverse = data.get('reverse', False)  # Whether to reverse (default is False)
    
    if not text or not target_language:
        return jsonify({"error": "Invalid request. Please provide both 'text' and 'target_language'."}), 400
    
    # Perform the translation
    translated_text = translate_text(text, target_language, reverse)
    
    return jsonify({"translated_text": translated_text})
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)       
