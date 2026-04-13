try:
    import fastapi
    print("fastapi ok")
    from fastapi import UploadFile
    print("UploadFile imported")
except ImportError as e:
    print(f"fastapi error: {e}")

try:
    import pydantic
    print("pydantic ok")
except ImportError as e:
    print(f"pydantic error: {e}")

try:
    import torch
    print("torch ok")
except ImportError as e:
    print(f"torch error: {e}")

try:
    import transformers
    print("transformers ok")
except ImportError as e:
    print(f"transformers error: {e}")

try:
    import nltk
    print("nltk ok")
except ImportError as e:
    print(f"nltk error: {e}")
