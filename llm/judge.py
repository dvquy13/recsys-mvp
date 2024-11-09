HISTORICAL_EVAL_PROMPT = """
Your task is to help me evaluate the relevance of the recommended item given the historical interactions of a user with some items.

Your output should be a JSON-compatible dictionary containing the following keys:
- rec_id: the ID of the recommendation
- is_relevant: 0 or 1
- reason: an explanation on why you give that relevance score

<EXAMPLE>

Example item titles with JSON input and output:

Input:
[
    {{
        "rec_id": 0,
        "historical_items": [
            'Minecraft - Xbox One',
            'Nintendo 2DS - Electric Blue with Mario Kart 7',
            'ReNext Tri-wing Screwdriver for Nintendo Wii, Gamecube, Gameboy Advance',
            'State of Decay 2 - Ultimate Edition - Xbox One',
            'Sleeping Dogs: Definitive Edition- PlayStation 4',
            'Xbox 360 Data Transfer Cable', 'Xbox 360 Play and Charge Kit',
            'Dragon Quest Builders - PlayStation 4',
            'DRAGON BALL Z: Kakarot - PlayStation 5', 'NBA Live 15 - Xbox One'
        ],
        "recommended_items": "inFAMOUS: Second Son Limited Edition (PlayStation 4)"
    }}
]


Output:  
[
    {{
        "rec_id": 0,
        "is_relevant": 1,
        "reason": "The user has a strong interest in action/adventure games and PlayStation titles, as shown by their history with 'Sleeping Dogs: Definitive Edition' and 'Dragon Quest Builders' on PlayStation consoles. 'inFAMOUS: Second Son' is also an action/adventure game on PlayStation 4, aligning well with the user's preferences."
    }}
]

</EXAMPLE>

Input:
{input_list}
Output:

"""
