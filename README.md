# LLMS
- Large Language Models (LLMs) are advanced AI system tranined on massive text
  datasets:
  - Understand
  - generate
  - process human language for tasks: (writing, summarizing, translating, answering question)

    --> Using deep learning (often Transformer architecture) to recognize complex patterns and produce human-like text.

- Fine-tuning data is the custom dataset you use to further train an existing AI model
  #### Simple definition
    - Fine-tuning data = input-output
  ##### What fine-tuning data look like
  Most fine-tuning datasets are structured as paris (or conversations)

  Example (Text model):
  ```bash
    {
        "input": "What is CI/CD?",
        "output": "CI/CD stands for Continuous Integration and Continuous Deployment..."
    }
  ```
  Example (Chat model):
  ```bash
    {
        "messages": [
            { "role": "user", "content": "Create a REST API in Node.js" },
            { "role": "assistant", "content": "Here is a simple Express REST API..." }
        ]
    }
  ```

### Requriement fine-tuning data 
- hugging face : we needs to create account for access to get model ai
- google colab : we needs to create account and for fine-tuning data model ai (Optinal: python3, R...)
    - Open google colab and Enable CPU
        1. go to google colab
        2. Runtime -> Change runtime type
        3. Hardware accelerator -> T4 GPU
        4. Save
    #### process fine-tuning data
    1. Import library
    ```bash
    !pip install -q -U transformers datasets accelerate peft trl bitsandbytes
    ```

    2. Install dependency 
    ```bash
    pip install pyarrow==19.0.0 --force-reinstall
    ```

    3. Login to Hugging Face
    ```bash
    from huggingface_hub import login
    login() # You will be prompted to enter your token
    ```
    - For login to huggin face so get model ai that traning data and then for get Token from huggin face for paste your HF access token (You must accept Gemma license on Hugging Face)
    - Access Tokens in access huggin face:
        - Token type -> Read
        - Token name (e.g. modelAi)
        - Create Token
        - Copy Token for pass token HF access login.

    4. For fine-tune model ai (gemma-2b-it)

    - Gemma 2B = Google's open-weight LLM (≈2 billion parameters)

    ```bash
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = "google/gemma-2b-it"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )

    print("✅ Model loaded successfully!")
    ```
    - We needs goto back account huggin face for get vertiy Model AI for run coding:
        - Search: gemma-2b-it select (google/gemma-2b-it)
        - Click: Acknownledge lincese
        - Click: Authorize

        After authorize ready we needs to create model ai Name:

        - Completed all field
        - Accept to contiunue 
        - After clcik run code and waiting process fine-tune

    
    


