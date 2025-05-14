You need to download this repo either with git clone https://github.com/kshitiz464/local_doc_qa.git or download zip folder.

Once you have the files, you need to open cmd or any terminal.

Inside the folder where all files are, you need to create virtual environment with python -m venv venv 
Then activate virtual env with :
  venv\Scripts\activate.bat
Run->  pip install -r requirements.txt  <-inside the project where all files are. 

Once they get installed, you need to download a LLM model from this page -> https://huggingface.co/models?library=gguf
Create models/ folder in main directory and save LLM model .gguf type in there. 
Update the model path in main.py line no 18. -> llm = Llama(model_path="models/gpt4all-falcon-q4_0.gguf", n_ctx=1024, n_threads=2)
change path to "model/your-model-name.gguf", you can increase n_ctx and n_threads if you have 16 gb ram, dedicated GPU and a good CPU. 
You also need to make changes in other places if you change n_ctx and n_threads for better performance.

Once these steps are done, you can just go in your terminal and run->  streamlit run main.py

Wait for sometime, and then the app will be ready to use in your browser at localhost:8501 port.
Upload the document you want, wait for processing, then ask whatever question you want. 

->> Contact me for help- 
kshitizyadav464@gmail.com
https://www.linkedin.com/in/kshitiz-yadav-209aba27a/
https://instagram.com/kshitiz464
Or open an issue in this git repo. 

Thanks! Please Follow my github if you liked the content!
