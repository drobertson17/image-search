<h1>Image Search</h1>

Categorizes, describes, and (one day) searches through a library of images.

<h2>Installation</h2>

<h3>Configure Models</h3>

There are two options for models: local and AWS Bedrock (or both).

For AWS Bedrock:
 1. Configure AWS account to access Bedrock via SSO. Some instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html). Be sure to name your profile `default`.
 2. Login to AWS with SSO: `aws sso login --profile default`

For local models:
 1. Download the models and place the files in the `/models/` directory. 
    - For the LLM, I used `Llama-3.2-3B-Instruct-Q5_K_S.gguf` from [here](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main).
    - For the VLM, I used `ggml-model-q4_k.gguf` and `mmproj-model-f16.gguf` from [here](https://huggingface.co/mys/ggml_llava-v1.5-7b/tree/main).
 2. Update the `.env` file to point to the correct files

<h3>Configure Database</h3>

A postgres database is required. Be sure to update the `.env` file accordingly. Example: `IMAGE_SEARCH_DATABASE="postgresql://user:pass@localhost/image_search"`

<h3>Build Environment</h3>

With python and poetry installed on the system, run `poetry install` from the repo directory.


<h2>Use</h2>

Update and run the `run.py` script in `/scripts/`
