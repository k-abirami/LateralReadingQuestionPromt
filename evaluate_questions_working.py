import csv
import json
import openai
import pandas as pd
from tqdm import tqdm

run_tag = 'Lateral-Reading-Rating-Test-Data-Evaluation'

input_file = 'test_data_without_labels.csv'
output_file = f'output/{run_tag}.csv'

data = pd.read_csv(input_file, sep='\t', header=None, names=['doc_id', 'run_tag', 'question'])

openai_client = openai.AzureOpenAI(
)

with open('rating_system_prompt.txt', 'r', encoding='utf-8') as file:
    system_prompt = file.read()

pbar = tqdm(total=len(data))

ratings = []

for idx, row in data.iterrows():
    doc_id = row['doc_id']
    question = row['question']

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',
         'content': f'Below is a question designed to evaluate the trustworthiness of an online news article through lateral reading.\n\n'
                    f'Question: "{question}"\n\n'
                    f'Please provide a rating from -1 to 4 on how effective this question is for lateral reading. '
                    f'Also, provide a brief explanation for your rating.'}
    ]

    try:
        response = openai_client.chat.completions.create(
            model='gpt-4-turbo-128k-20240409',
            temperature=0,
            presence_penalty=0,
            top_p=1,
            frequency_penalty=0,
            messages=messages,
        )

        gpt_response = response.choices[0].message.content.strip()
        lines = gpt_response.split('\n')
        rating = next((line.split(':')[-1].strip() for line in lines if 'Rating' in line), None)
        explanation = '\n'.join(lines[1:]).strip()


        ratings.append([doc_id, run_tag, question, rating])

    except openai.BadRequestError as error:
        print(f'[ERROR] {error.message}')
        ratings.append([doc_id, run_tag, question, 'Error', 'An error occurred during processing.'])

    pbar.update(1)

pbar.close()

output_df = pd.DataFrame(ratings, columns=['doc_id', 'run_tag', 'question', 'rating'])
output_df.to_csv(output_file, index=False, encoding='utf-8', sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\')

print(f'Ratings saved to {output_file}')