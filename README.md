# Process quiz data from GCP Vertex AI responses

### Jupyter Notebook file: [VIEW](quiz_data_processing.ipynb)

## Introduction

In this project, we will build a Jupyter notebook to create a synthetic quiz dataset for using in web app.

Quiz data could be generated from various open sources. These sources provide quizzes gathered from user's posting.
Considering this nature and be afraid of content right, I wanted to try other method of getting fake, or synthetic data.

As I'm dedicated to using cloud solutions in [Google Cloud Platform (GCP)](https://cloud.google.com/),
I get my solution from [Vertex AI](https://cloud.google.com/vertex-ai).

## Preparing

### 1. Import, authenticate and assign

We need to import Pandas and some other general packages.

To make GCP jobs work out, we authenticate GCP credentials
using environment variable `GOOGLE_APPLICATION_CREDENTIALS`. It's also convenient to assign constants related to GCP service.

```
# Import general packages
import pandas as pd
import re
import io
import json

# Set environment variable to authenticate GCP credentials
!export GOOGLE_APPLICATION_CREDENTIALS='book-to-quiz-7558e7ee5aca.json'

LOCATION = "us-central1"
PROJECT = 'book-to-quiz'
BUCKET = 'book-to-quiz-question-bank'

MODEL = "gemini-1.5-flash-001"
```

### 2. Define functions interact with GCP
It's better to follow GCP docs and prepare some functions to work with GCP so that we can use later with ease.

For Google Cloud Storage (GCS), we make `gcs_read` and `gcs_write` to read from and write to.

```
# Imports the Google Cloud client library
from google.cloud import storage

def gcs_read(bucket_name, blob_name):
    """Read a blob from GCS using file-like IO"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    with blob.open("r") as file:
        # Return data as lines
        return file.readlines()
    
def gcs_write(bucket_name, blob_name, content):
    """Write a blob from GCS using file-like IO"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    with blob.open("w") as file:
        file.write(content)
```

For Vertex AI, we need a `generate` as a generator to yield response in processing.

```
# Import Vertex AI packages
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

def generate(p_text, g_config, s_settings):
    """Yield text with generator which receive prompt text, generation config and safety settings as arguments"""
    vertexai.init(project=PROJECT, location=LOCATION)
    model = GenerativeModel(
        MODEL,
    )
    responses = model.generate_content(
        [p_text],
        generation_config=g_config,
        safety_settings=s_settings,
        stream=True,
    )

    for response in responses:
        yield response.text
```

## Generate Vertex AI content

Define variables with value we will use for Vertex AI.

```
prompt = """Generate quiz with these requirements:
- Total questions: 50.
- Topic: AWS.
- Difficulty: Hard.
- Types: true/false, single correct answer, multiple correct answers.
- Indent questions with number.
- Indent choices and true/false with upper letter.
- If more than 2 correct answers questions, remind  \"(select [exact number] apply)\" before choice A.
- In each question, total correct answers is less than total choices.
- Choices consist multiple technically complicated steps.
- Show correct choices at the end of each question.
- No markdown, plain text.
- Group by type."""

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
```

Assign generator to a variable.

```
# Assign generator to a variable
generator = generate(prompt, generation_config, safety_settings)
```

Call the generator and concatenate text from generator's responses together to make the full text.

```
# Use string comprehension to gather text from generator
generate_text = ''.join(response for response in generator)
```

## Export generated data to file, write to GCS

Extract topic, difficulty and size from prompt text. Combine all into a TXT file name.

```
# Read generated text as lines
buf = io.StringIO(prompt)
lines = buf.readlines()

# Check topic line
topic_line = [line for line in lines if '- Topic: ' in line][0]
topic = topic_line[9:].rstrip('\n.')
topic = re.sub(' ', '_', topic.lower())
print('Topic:', topic)

# Check difficulty line
difficulty_line = [line for line in lines if '- Difficulty: ' in line][0]
difficulty = difficulty_line[14:].rstrip('\n.').lower()
print('Difficulty:', difficulty)

# Check size line
size_line = [line for line in lines if '- Total questions: ' in line][0]
size = size_line[19:].rstrip('\n.')
print('Size:', size)

# Combine topic, difficulty and size into .txt file name
file_name = f'{topic}_{difficulty}_{size}.txt'
print('File name:', file_name)
```

Write the generated data to our GCS bucket with prepared file name in TXT format.

```
# Assign file_name to a constant using for writing to GCS
WRITE_BLOB = file_name

gcs_write(BUCKET, WRITE_BLOB, generate_text)
```

## Load data from GCS

We will use the same file we've just exported for the purpose of loading data from GCS.

```
# Assign file_name to a constant using for reading from GCS
READ_BLOB = file_name

data = gcs_read(BUCKET, READ_BLOB)

# Use display() because data is already list of lines structure
display(data)
```

## Process data

### 1. Split questions and answers from text

Define function `split_qa`.

```
def split_qa(lines_data):
    """Accepts list of lines. Returns a dictionary with keys `question` and `answer`"""
    dict_data = {'question': [],
                 'answer': []}
    
    # Switch determines if the last line is in question section or not
    q_prev = False
    
    for line in lines_data:
        # Strip '**' style around 'Correct answer(s)'
        line = line.replace('**', '')
        # Check if line is not blank or title, heading ('##')
        if (line != '\n') and ('##' not in line):
            # Call the first word of the line is `head`
            head = line.split()[0]
            
            # Append new question if all of these meet:
            # - `q_prev` is False
            # - First character of `head` is numeric
            # - Last character of `head` is '.'
            if (not q_prev) and head[0].isnumeric() and head[-1] == ".":
                dict_data['question'].append(line)
                q_prev = True
    
            # Append new answer if all of these meet:
            # - `q_prev` is True
            # - `head` is 'A.'
            elif q_prev and head == 'A.':
                dict_data['answer'].append(line)
                q_prev = False
    
            # Add line to unfinished question
            elif q_prev:
                dict_data['question'][-1] += line
        
            # Add line to unfinished answer
            else:
                dict_data['answer'][-1] += line
    
    return dict_data
```

Call the function and convert the result into DataFrame class.

```
# Return split_qa result to a variable
dict0 = split_qa(data)

# Convert `dict0` to a DataFrame
df0 = pd.DataFrame(dict0)

print('Number of questions:', len(df0))
print('Columns:', df0.columns.values)
```

### 2. Tidy question text

Tidy question text by stripping indentation and new lines character.

```
# Make a copy of df0
df1 = df0.copy()
```
```
# Strip the number at the beginning and '\n' at the end of each question 
df1['question'] = df1['question'].str.replace(r'^\d{0,4}\.[ ]', '', regex=True).str.rstrip()
```

### 3. Split answer text into multiple choices

Define `split_choice` to split answer text into multiple choices.
Choices will be confirmed as correct and incorrect thereafter.

```
def convert_index(capital):
    """Return zero-based index from capital, 'A' has unicode code as 65"""
    return ord(capital) - 65

def split_choice(answer):
    """
    Split the `answer` data into multiple choices
    """
    # print(answer)
    
    # Use 'Correct Answer(s): ' to split text.
    # Index 0 is all choices, index 1 is all answers
    split_all = re.split(r'Correct Answer[s]*: ', answer)
    # print("After strip 'Correct Answer: ': ", split_all)
    
    choices = split_all[0]
    # print('Choices text:', choices)
    
    correct_stack = split_all[1]
    # print('Correct stack:', correct_stack)
    
    # Split using ','
    correct_stack = correct_stack.split(',')
    
    # Pick only first capital indicating the choices
    correct_note = [item.strip()[0] for item in correct_stack]
    # print('Correct note:', correct_note)
    
    # Make zero-based index from alphabet
    correct_index = [convert_index(item) for item in correct_note]
    # print('Correct index', correct_index)
    
    # Split using 'X. ', index 0 is '', so pass
    choices = re.split(r'[A-Z]\.[ ]', choices)[1:]
    # Strip right side of choice text
    choices = [choice.rstrip() for choice in choices]
    # print('All choices in list: ', choices)
    
    incorrect = [choice for index, choice in enumerate(choices) if index not in correct_index]
    # print('Incorrect in list: ', incorrect)
    correct = [choice for index, choice in enumerate(choices) if index in correct_index]
    # print('Correct in list: ', correct)
    
    return {'incorrect': incorrect,
            'correct': correct}
```

Apply `split_choice` to dataset and store choices in new columns, named `incorrect` and `correct`. 

```
# Make a copy of `df1`
df_2 = df1.copy()
```
```
# Apply the function to full dataset
df_2['incorrect'] = df_2['answer'].apply(split_choice).str['incorrect']
df_2['correct'] = df_2['answer'].apply(split_choice).str['correct']
```