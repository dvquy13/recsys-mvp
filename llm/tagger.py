PROMPT = """
For each of the following items, generate a set of tags that capture the main features, target audience, platform compatibility, and item type (e.g., accessory, game, hardware).

Return the output as a list of JSON objects, where each object includes the original item title and a list of tags. Focus on adding tags that help shoppers easily find these items based on gaming platforms, brand associations, special edition details, and product functionality.

<EXAMPLE>

Example item titles with JSON input and output:

Input:  
[
    {{ "title": "Gliging 120Pcs/Set MX Switch Films Mechanical Keyboard Switches stabilizer Switch Film Repair for Cherry MX kailh Gateron Switch" }},
    {{ "title": "NEW HOLLAND SKYLINE [Xbox 360]" }},
    {{ "title": "HORI Gaming Headset (Pikachu POP) for Nintendo Switch & Switch Lite - Officially Licensed by Nintendo & Pokemon Company International - Nintendo Switch" }}
]

Output:  
[
    {{
        "item_title": "Gliging 120Pcs/Set MX Switch Films Mechanical Keyboard Switches stabilizer Switch Film Repair for Cherry MX kailh Gateron Switch",
        "tags": ["Keyboard Accessory", "Cherry MX", "Mechanical Keyboard", "Switch Film", "Gateron", "Repair"]
    }},
    {{
        "item_title": "NEW HOLLAND SKYLINE [Xbox 360]",
        "tags": ["Xbox 360 Game", "Racing", "NEW HOLLAND", "Skyline", "Retro"]
    }},
    {{
        "item_title": "HORI Gaming Headset (Pikachu POP) for Nintendo Switch & Switch Lite - Officially Licensed by Nintendo & Pokemon Company International - Nintendo Switch",
        "tags": ["Headset", "Nintendo Switch", "Switch Lite", "Pikachu", "Pokemon", "Gaming Accessory"]
    }}
]

</EXAMPLE>

Input:
{input_list}
Output:
"""
