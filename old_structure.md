MeetingMind-AI-Redesigned/
├── __pycache__/
├── .venv/
├── .vscode/
├── github_raw_data/            # Raw data collected from GitHub
|   ├── AMICorpusXML/   
│   ├── GoogleData (MISed)/     # (Using this for labelled dataset)
|   ├── MeetingBank/  
├── ml_backend/                 # Machine Learning components
|   ├── requirements.txt   
|   ├── __init__.py            
│   ├── __pycache__/   
│   ├── api/
|   |    ├── __pycache__/
|   |    ├── routes/
|   |        ├── __pycache__/
|   |        ├── __init__.py
|   |        ├── entities.py
|   |        ├── summarize.py
|   |        ├── transcribe.py
|   |    ├── __init__.py
|   |    ├── main.py
|   ├── dataset/                  # Processed and labelled CSV dataset
|   |    ├── generate_dataset.py  # Script to generate training datasets
|   |    ├── labelled_data.csv    # Labelled data for training ML model (Text transcripts)
│   ├── models/                   # Model architectures and utilities        
|   |   ├── __pycache__/
|   |   ├── __init__.py 
│   │   ├── bart_summarizer.py
|   |   ├── bert_ner.py
│   │   ├── whisper_finetune.py
│   ├── preprocessing/            # Data cleaning and text processing
|   |   ├── __pycache__
│   │   ├── __init__.py
│   │   └── text_cleaner.py
│   └── training/                 # Model training notebooks and scripts
│       ├── whisper-finetuned/
│       ├── train_summarizer.ipynb
│       ├── train_whisper.ipynb
|       ├── evaluate_models.ipynb
│       ├── __init__.py            # Saved model checkpoints
└── src/                           # Core application logic
|    ├── __pycache              
|    ├── gemini_layer.py         # Google Gemini API integration
|    ├── insight_extractor.py    # Logic for meeting insight extraction
|    └── __init__.py
├── .env
├── .gitignore
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and secret management
├── labelling_guidelines.md     # Guidelines for data labelling
├── pyrightconfig.json          # Static type checking configuration
├── raw_data_format.md          # Documentation for raw data structure
├── requirements.txt            # Python dependencies
├── ui_preview.html             # UI Mockup/Preview