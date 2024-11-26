import csv
import json
import openai
import pandas as pd
from tqdm import tqdm


run_tag = 'Organizers-Baseline-5'
# V1: Without the example questions
# V2: With the example questions

data = []
with open('trec-2024-lateral-reading-task1-one-article.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        record = json.loads(line.strip())
        data.append([record['ClueWeb22-ID'], record['Clean-Text']])


openai_client = openai.AzureOpenAI(
    
)


#system_prompt = open('system_prompt_v2.txt', 'r').read()

with open('system_prompt_v4.txt', 'r', encoding='utf-8') as file:
    system_prompt = file.read()


output = []
pbar = tqdm(total=len(data))
for i in range(1):
    doc_id = data[i][0]
    document = data[i][1]

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',
         'content': f'Below is an online news article. \n\n'
                    f'---BEGIN NEWS ARTICLE--- \n\n'
                    f'{document} \n\n'
                    f'---END NEWS ARTICLE--- \n\n'
                    f'Please come up with 10 questions to help the reader evaluate the trustworthiness of the above '
                    f'news article. Each question should be at most 120 characters long. '
         }
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

        gpt_response = response.choices[0].message.content

        for j, line in enumerate(gpt_response.strip().split('\n')):
            period_index = line.find('. ')
            if period_index != -1:
                question = line[period_index + 2:].strip()
                if len(question) > 120:
                    print(f'[WARNING] Article {i} Question {j + 1}: {question} exceeds 120 characters.')
                output.append([doc_id, run_tag, j + 1, question])

    except openai.BadRequestError as error:
        print(f'[Error] {error.message}')
        if error.code == 'content_filter':
            print(f'[ERROR] Article {i}: {error.message}')
        else:
            print('f[ERROR]')

    pbar.update(1)


output = pd.DataFrame(output, columns=['doc_id', 'run_tag', 'rank', 'question'])
output.to_csv(f'output/{run_tag}', index=False, encoding='utf-8', sep='\t',
              quoting=csv.QUOTE_NONE, escapechar='\\', header=False)
