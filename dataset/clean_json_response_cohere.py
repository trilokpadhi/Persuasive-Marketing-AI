import json
import pandas as pd

path = "/staging/users/tpadhi1/Persuasive-Marketing-AI/output.txt"

"""
This script is used to clean the JSON response from the CoHere API and save it to a file.

json_string = """
{
  "headline": "100 Most Anticipated Books Releasing in 2010",
  "num_of_strategies": 6,
  "strategies": [
    "Social Proof",
    "Authority",
    "Trustworthiness",
    "Scarcity",
    "Reciprocity",
    "Concreteness"
  ],
  "reasons": "The headline leverages social proof by implying that these books are highly anticipated by many, which influences readers to take interest. Authority and trustworthiness are established by presenting a list of anticipated books, positioning the source as a credible authority in the literary world. Scarcity is implied through the specific reference to '2010', creating a sense of urgency to discover these anticipated releases. Reciprocity is also implied as the headline offers valuable information to readers who are eager to find new books to read in the coming year. Lastly, concreteness is used by providing a specific number, making the headline more memorable and shareable."
}
"""

# Remove leading/trailing whitespace and newline characters
json_string = json_string.strip()

# Convert the JSON string to a Python dictionary
data = json.loads(json_string)

# Now you can access the data like a normal dictionary
print(data["headline"])
"""

with open(path, 'r') as f:
    for line in f:
        json_string = line.strip()
        data = json.loads(json_string)
        print(data["headline"])
        print(data["num_of_strategies"])
        print(data["strategies"])
        print(data["reasons"])
        print("\n")
        # write the data to a json file 
