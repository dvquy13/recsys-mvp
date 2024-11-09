SYSTEM_PROMPT = "You are a helpful assistant and an expert in Video Game industry."

PROMPT = """
For each of the following items, generate a set of tags that capture the main features, target audience, platform compatibility, and item type (e.g., accessory, game, hardware).

Return the output as a list of JSON objects, where each object includes the original item `title`, `parent_asin`, and a list of tags.

Focus on adding tags that help shoppers easily find these items based on genre, brand associations, and product functionality.

If possible, add new information that is not obvious from the item title.

Do not include the tags that are similar to those in the input categories, for example do not include tags about gaming platforms as output.

<EXAMPLE>

Example item titles with JSON input and output:

Input:  
[
    {{ "parent_asin": "B09ZYFYT5J", "title": "Gliging 120Pcs/Set MX Switch Films Mechanical Keyboard Switches stabilizer Switch Film Repair for Cherry MX kailh Gateron Switch" }},
    {{ "parent_asin": "B00GZ6TX7E", "title": "NEW HOLLAND SKYLINE [Xbox 360]" }},
    {{ "parent_asin": "B09QZ2RZLV", "title": "HORI Gaming Headset (Pikachu POP) for Nintendo Switch & Switch Lite - Officially Licensed by Nintendo & Pokemon Company International - Nintendo Switch" }}
]

Output:  
[
    {{
        "parent_asin": "B09ZYFYT5J",
        "item_title": "Gliging 120Pcs/Set MX Switch Films Mechanical Keyboard Switches stabilizer Switch Film Repair for Cherry MX kailh Gateron Switch",
        "tags": ["Keyboard Accessory", "Cherry MX", "Mechanical Keyboard", "Switch Film", "Gateron", "Repair"]
    }},
    {{
        "parent_asin": "B00GZ6TX7E",
        "item_title": "NEW HOLLAND SKYLINE [Xbox 360]",
        "tags": ["Racing", "NEW HOLLAND", "Skyline", "Retro"]
    }},
    {{
        "parent_asin": "B09QZ2RZLV",
        "item_title": "HORI Gaming Headset (Pikachu POP) for Nintendo Switch & Switch Lite - Officially Licensed by Nintendo & Pokemon Company International - Nintendo Switch",
        "tags": ["Headset", "Nintendo Switch", "Switch Lite", "Pikachu", "Pokemon", "Gaming Accessory"]
    }}
]

</EXAMPLE>

Input:
{input_list}
Output:
"""
