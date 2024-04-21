
# !pip install cohere

import json
import cohere

api_key="sgEAXCP80PTxsdY4pg5sI6qBrR1M6xtvwERMDl2K"
model = "command-r-plus"
filename = '/content/knowledgebase.json'

def add_books_to_json(prompt, model, responses_list, next_id):
    co = cohere.Client(api_key)

    try:
        # Make the API call
        response = co.chat(message=prompt, model=model)
        res = response.text
        res = res.replace('json', ' ').replace('```', ' ')
        print(res)

        # Check if the response has text and is not an error
        if response.text:
            # Parse the JSON-formatted text into dictionaries
            books_list = json.loads(res)
            # Iterate over each book and add additional information
            for book in books_list:
                # Create a new dictionary that includes the extra details
                enhanced_book = {
                    'id': next_id,
                    'prompt': prompt,
                    'model': model,
                    **book  # This unpacks the existing dictionary into the new one
                }
                responses_list.append(enhanced_book)
                next_id += 1  # Increment the ID for the next book
        else:
            print("No data received from the API for prompt:", prompt)
    except (json.decoder.JSONDecodeError, Exception) as e:
        print("Error occurred:", e)
        print("Failed to process prompt:", prompt)
    return next_id  # Return the updated ID counter

# Example prompts
prompts = [
    "Tell me top 10 advertising books with 10 lines from the book which can be used as marketing advertisement. I need the results precisely in json format with the title and list of texts addressing advertisements, nothing else",
    "Tell me top 10 advertising papers with 10 lines from the book which can be used as marketing advertisement. I need the results precisely in json format with the title and list of texts addressing advertisements, nothing else",
    "Tell me top 10 advertising websites with 10 lines from the book which can be used as marketing advertisement. I need the results precisely in json format with the title and list of texts addressing advertisements, nothing else",
    "Tell me top 10 persuasion methods books with 10 lines from the book which can be used as marketing advertisement. I need the results precisely in json format with the title and list of texts addressing advertisements, nothing else",
    "Tell me top 10 persuasion methods papers with 10 lines from the book which can be used as marketing advertisement. I need the results precisely in json format with the title and list of texts addressing advertisements, nothing else",
    "Tell me top 10 persuasion methods websites with 10 lines from the book which can be used as marketing advertisement. I need the results precisely in json format with the title and list of texts addressing advertisements, nothing else"
]


responses = []
next_id = 1  # Start ID counter from 1

for prompt in prompts:
    next_id = add_books_to_json(prompt, model, responses, next_id)

# Write the responses to a JSON file
filename = "responses.json"
with open(filename, 'w') as json_file:
    json.dump(responses, json_file, indent=4)

print("Responses:", responses)

print(len(responses))