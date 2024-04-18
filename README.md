This project is about Retreive, Rerank, and Generate.

## Project Structure
`dataset/`: contains the dataset used in this project.


## Models
`retriever/`: contains the retriever model.

## Notes
- The model was generating garbage, it was solved by using the `--no_repeat_ngram_size 2` flag, instrcution tokens and using the chat model.