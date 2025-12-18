# LLMS
- Large Language Models (LLMs) are advanced AI system tranined on massive text
  datasets:
  - Understand
  - generate
  - process human language for tasks: (writing, summarizing, translating, answering question)

    --> Using deep learning (often Transformer architecture) to recognize complex patterns and produce human-like text.

- Fine-tuning data is the custom dataset you use to further train an existing AI model
- datasets.json is data education-related records for Cambodia:
  - Topic: Eudcation Cambodia
    - Ministry of Education (MoEYS)
    - School quality control
    - Teacher qualifications
    - Student enrollment
    - Curriculum standards
    - School inspection
    - Provinces across Cambodia 

  ### Simple definition
    - Fine-tuning data = input-output
  #### What fine-tuning data look like
  Most fine-tuning datasets are structured as paris (or conversations)

  ### Fine-tuning Process
  ![Alt text](https://cdn.prod.website-files.com/5ee50f2ef83ac07f0cb7fb44/668ac22b600ee297e294c0f4_6564bfb0531ac2845a2562f3_Finetuning_process_49bc08a9e9.jpeg)

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

  1, Hugging face 

  2, Google Colab

  3, Wandb.ai (Make sure change Region for Cambodia must to change use VPN)


- hugging face : we needs to create account for access to get model ai
- google colab : we needs to create account and for fine-tuning data model ai (Optinal: python3, R...)
- wandb.ai : we need to create account is AI developer platform, also known by its domain wandb.ai, that provides     tools for the entire machine learning lifecycle, focusing on experiment tracking, model management, dataset versioning, and collaboration to help teams build, debug, and deploy AI models faster

   - Open google colab and Enable CPU
        1. go to google colab
        2. Runtime -> Change runtime type
        3. Hardware accelerator -> T4 GPU
        4. Save
    ### process fine-tuning data
  #### STEP 1. Import library
    ```bash
    !pip install -q -U transformers datasets accelerate peft trl bitsandbytes
    ```

  #### STEP 2. Install dependency 
    ```bash
    pip install pyarrow==19.0.0 --force-reinstall
    ```

  #### STEP 3. Login to Hugging Face
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

  #### STEP 4. For fine-tune model ai (gemma-2b-it)

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
        - After clcik run code and waiting process download model ai (gemma-2b-it)


  #### STEP 5. Upload datasets in file
    - Right Click: New File
    - Name file: (e.g. datasets.json)
    We will fine-tuning datasets.json


  #### STEP 6. Load Datasets
    ```bash
    from datasets import load_dataset

    dataset = load_dataset("json", data_files="dataset.json", split="train")
    ```
    - Import datasets into our code
    

  #### STEP 7. Config LoRa
    - LoRa config (Low-Rank Adaptation configuration) defines how your base AI Model is fine-tuned
    
    ```bash
    from peft import LoraConfig

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    ```

  #### STEP 8. Fine Tuning

    ```bash
    from transformers import TrainingArguments
    from trl import SFTTrainer

    def formatting_func(example):
        return f"User: {example['instruction']}\nAssistant: {example['response']}"

    training_args = TrainingArguments(
        output_dir="phearadevAi",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    trainer.train()

    ```
    - We can change your own model ai
      - Output_dir="phearadevAi" ->  output_dir="exmple name.."
      - After fine-tune and then save folder ai 
      - And to needs api-key Wandbai
    - Wandbai must to create account (we needs change Region use VPN for country Cambodia must to change cuz it not supported)
      - After completd all field 
      - Choice Models Train, fine-tune, and manage AI models
      - Copy Api-key for loggin the wandb library (e.g. 9837234....)
      - And then pass the api-key is ready 
      - Please waiting for fine-tuning data...
      
    - Then it's have 2 Folder:
      - Folder 1 name model ai your own
      - Folder 2 wandb for connect into api


  #### STEP 9. Testing Chat

    ```bash
    prompt =  "How does Cambodia ensure education quality in public schools?\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=150)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    ```

  #### Noted: If correct the data make sure in dataset.json must to be is have manay data field.

    
  #### STEP 10. Push code to Hugging Face

    ```bash
    from huggingface_hub import login

    # Paste your token when prompted
    login()
    ```

    - Need token for loggin to huggin face space
      - Goto account hugging face
      - Access Tokens
      - Create a new token
      - Token type -> Write (Make sure is Write)
      - Token name: (e.g. pushModelAi)
      - Copy Token
      - Pass to HF access tokens

    
  #### STEP 11. Repo name

    ```bash
    # Replace "your-hf-username" and "my-gemma-model" with your details
    repo_name = "phearaa/myAi"

    # Push the model to the Hub
    model.push_to_hub(repo_name)

    # Push the tokenizer to the Hub
    tokenizer.push_to_hub(repo_name)

    print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_name}")
    ```

    - Change repo your own: repo_name = "phearaa/myAi" 
      - For: phearaa is username of hugging face
      - After push code running
      - U can check in profile hugging face for model ai
      - Please waiting psuh model successfully.

    - For account wandb you can check you look data conversation
  #### Trip: We can create space save model
    - We can deploy with service: aws, google cloude..

## GET API Fine-Tuned Data
- When get fastAPi from colab model for POST in web app localhost owner
  - How it works
  ```bash
    [Frontend Web App]
        |
        | HTTP request (POST /chat)
        v
  [API Server in Colab (FastAPI)]
        |
  [Gemma 2B + LoRA fine-tuned model]
        |
        v
      Response
  ```

  - Frontend: Gradio
  - Backend: API is FastAPI running in Colab
  - The API receives a message -> returns a response from your fine-tuned model

  #### STEP 1. Install required package in Colab
  
  ```bash
  !pip install transformers peft bitsandbytes torch fastapi uvicorn nest-asyncio pyngrok
  ```

  #### STEP 2. Load your fine-tuned model
  ```bash
  from transformers import AutoTokenizer, AutoModelForCausalLM
  from peft import PeftModel

  base_model = "google/gemma-2b"
  lora_path = "./gemma-edu-lora"  # your fine-tuned LoRA folder

  tokenizer = AutoTokenizer.from_pretrained(base_model)
  model = AutoModelForCausalLM.from_pretrained(
      base_model,
      load_in_8bit=True,
      device_map="auto"
  )

  model = PeftModel.from_pretrained(model, lora_path)
  model.eval()
  ```

  #### STEP 3. Create API with FastAPI

  ```bash
  from fastapi import FastAPI
  from pydantic import BaseModel

  app = FastAPI()

  class ChatRequest(BaseModel):
      message: str

  @app.post("/chat")
  def chat(req: ChatRequest):
      prompt = f"<start_of_turn>user\n{req.message}<end_of_turn>\n<start_of_turn>model\n"
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
      outputs = model.generate(**inputs, max_new_tokens=200)
      response = tokenizer.decode(outputs[0], skip_special_tokens=True)
      return {"response": response}
  ```  

  - Endpoint: POST/chat
  - Payload: {"message": "Your question"}
  - Response: {"response": "AI answer"}

  #### STEP 4. Expose API to the Web

    - Colab VMs are private, so use ngrok:
  
  ```bash
  import nest_asyncio
  from pyngrok import ngrok
  import uvicorn

  nest_asyncio.apply()
  public_url = ngrok.connect(8000)
  print("Public URL:", public_url)

  uvicorn.run(app, host="0.0.0.0", port=8000)

  ```

  - public_url is now your API URL (e.g., http://abced1234.ngrok.io)
  - Frontend can send POST request to http://abced1234.ngrok.io/chat

  #### STEP 5. Connect Fronted to API
    - Python Gradio example:

  ```bash
  import gradio as gr
  import requests

  def chat_ui(message):
      url = "http://abcd1234.ngrok.io/chat"
      res = requests.post(url, json={"message": message})
      return res.json()["response"]

  gr.Interface(chat_ui, gr.Textbox(), gr.Textbox()).launch()
  ```

## Project Structure API + Interface

  ```bash
      education-chat-app/
    │
    ├── model/                    # Fine-tuned model + LoRA adapters
    │   └── gemma-edu-lora/       # Copy your LoRa fine-tuned model here
    │
    ├── app/                      # Backend logic
    │   ├── __init__.py
    │   ├── model_loader.py       # Load your model and tokenizer
    │   ├── chat.py               # Chat function calling model
    │   └── api.py                # FastAPI endpoints (POST /chat)
    │
    ├── ui/                       # Frontend interface
    │   ├── gradio_ui.py          # Python Gradio interface (optional)
    │   └── web/                  # Optional HTML/JS frontend
    │       ├── index.html
    │       └── script.js
    │
    ├── data/                     # Dataset files (optional)
    │   ├── datasets.json
    │   └── datasets.jsonl
    │
    ├── notebooks/                # Colab notebooks
    │   └── fine_tune.ipynb
    │
    ├── requirements.txt          # All Python dependencies
    ├── README.md
    └── run.py                    # Entry point to run Gradio or FastAPI
  ```
  
  #### Description function Folder/File

  1, model/ -> fine-tuned Gemma 2B + LoRa adpters

  2, app/   -> backend logic (model loader, chat, API)

  3, ui/    -> frontend interface (Gradio)

  4, data/  -> datasets files (optional)
  
  5, notedbooks/  -> Colab notebooks for fine-tuning/testing


  #### Write the code here
  
  ##### 1. model_leader.py
  ```bash
  from transformers import AutoTokenizer, AutoModelForCausalLM
  from peft import PeftModel

  base_model = "google/gemma-2b"
  lora_path = "./model/gemma-edu-lora"

  tokenizer = AutoTokenizer.from_pretrained(base_model)
  model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=True, device_map="auto")
  model = PeftModel.from_pretrained(model, lora_path)
  model.eval()
  ```

  ##### 2. chat.py
  
  ```bash
  from app.model_loader import model, tokenizer

  def chat_with_ai(message):
      prompt = f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n"
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
      outputs = model.generate(**inputs, max_new_tokens=200)
      response = tokenizer.decode(outputs[0], skip_special_tokens=True)
      return response
  ```

  ##### 3. api.py
  
  ```bash
  from fastapi import FastAPI
  from pydantic import BaseModel
  from app.chat import chat_with_ai

  app = FastAPI()

  class ChatRequest(BaseModel):
      message: str

  @app.post("/chat")
  def chat(req: ChatRequest):
      return {"response": chat_with_ai(req.message)}
  ```

  ##### 4. gradio_ui.py

  ```bash
  import gradio as gr
  from app.chat import chat_with_ai

  iface = gr.Interface(
      fn=chat_with_ai,
      inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),
      outputs=gr.Textbox(label="AI Response"),
      title="Education AI Chat"
  )

  iface.launch()
  ```

  ##### 5. run.py
  
  ```bash
  from ui.gradio_ui import iface

  if __name__ == "__main__":
      iface.launch()
  ```

  ##### 6. Create requirements.txt

  ```bash
  transformers
  peft
  bitsandbytes
  torch
  gradio
  fastapi
  uvicorn
  nest-asyncio
  pyngrok
  ```

  - Later, you can run:
  
  ```bash
  pip install -r requirements.txt
  ```

  ##### 7. Run your project
  - Option 1: Gradio Interface

    ```bash
    python run.py
    ``` 
  
  - Option 2: FastAPI API

    ```bash
    uvicorn app.api:app --reload
    ```

      - API runs at http://127.0.0.1:8000/chat
      - Frontend can send POST requests here
  

## License
This project is licensed under the [MIT License](LICENSE).
    
    


