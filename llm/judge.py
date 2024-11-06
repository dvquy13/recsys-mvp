PROMPT = """
You are a helpful assistant and an expert in Video Game industry.

Your task is to help me evaluate the relevance of the recommendations given the historical interactions of a user with some items.

Your output should be a JSON-compatible dictionary containing the following keys:
- rec_id: the ID of the recommendation
- relevant_score: range [0, 5] where 5 is the most relevant and 0 is the least relevant
- reason: an explanation on why you give such relevant score
- tags: summarize the above reasons into list of keywords or tags. We need this information to group the examples to identify which categories we are making errors with and from that able to brainstorm new features to help the recommendation to learn better.

Here are the user historical interactions:
1. DotA 2
2. CS:GO
3. League of Legends

Here are the recommendations:
1. Pokemon Go
2. Call of Duty: Warzone
3. Mobile Legends: Bang Bang
4. XBox controller
5. Microsoft SD card
"""


PROMPT = """
You are a helpful assistant and an expert in Video Game industry.

Your task is to help me label the users based on their interactions to identify which keywords can describe the user's interest.

Your output should be a JSON-compatible dictionary containing the following keys:
- user_id: the ID of the user
- keyword: a keyword describing the user's interest
- reason: an explanation on why you give such relevant score
- tags: summarize the above reasons into list of keywords or tags. We need this information to group the examples to identify which categories we are making errors with and from that able to brainstorm new features to help the recommendation to learn better.

Here are the user historical interactions:
1. DotA 2
2. CS:GO
3. League of Legends
"""
