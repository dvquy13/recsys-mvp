import gradio as gr
import pandas as pd
from api_call import get_recommendations, get_user_features, push_new_item_sequence
from loguru import logger
from query import get_items_metadata, get_users
from theme import blueq

# Fetch users
users = get_users()
sample_users = users[:10]


# Define the function that will use the selected dropdown value
def _get_items_df(selected_user):
    features = get_user_features(selected_user)
    logger.info(f"Features: {features}")
    _idx = features["metadata"]["feature_names"].index(
        "user_rating_list_10_recent_asin"
    )
    item_sequence_str = features["results"][_idx]["values"][0]
    item_sequences = item_sequence_str.split(",")
    items_metadata = get_items_metadata(item_sequences)
    items_metadata = (
        items_metadata.set_index("item_id")
        .loc[item_sequences[::-1]]
        .reset_index()
        .reset_index()
        .rename(columns={"index": "recency_rank"})
    )

    # Return the entire DataFrame to be displayed in a table format
    return items_metadata.head(10)


def _get_recs(selected_user):
    # Get recommendations for the selected user
    recommendations = get_recommendations(selected_user, debug=False)
    recs_df = pd.DataFrame(recommendations["recommendations"])
    rec_item_ids = recs_df["rec_item_ids"].values.tolist()
    items_metadata = get_items_metadata(rec_item_ids)
    output = pd.merge(
        recs_df, items_metadata, how="left", left_on="rec_item_ids", right_on="item_id"
    )
    return output


def process_likes(*responses):
    df = responses[-2]
    user_id = responses[-1]
    logger.info(f"{user_id=}")
    logger.info(f"{df=}")
    responses = responses[:-2]
    liked_items = []
    disliked_items = []
    item_sequences = []

    for i, r in enumerate(responses):
        title = df.iloc[i]["title"]
        item_id = df.iloc[i]["item_id"]
        item_display = f"{title} ({item_id})"
        if r == "👍":
            liked_items.append(item_display)
            item_sequences.append(item_id)
        if r == "👎":
            disliked_items.append(item_display)
            item_sequences.append(item_id)

    push_new_item_sequence(user_id, item_sequences)

    return f"You liked: {liked_items}\nYou disliked: {disliked_items}"


# Create the Gradio interface with Inter font and Roboto Mono as monospace font
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=blueq,
        font=["Inter", "sans-serif"],
        font_mono=["Roboto Mono", "monospace"],
    )
) as demo:
    gr.Markdown("# RecSys MVP Demo")

    # Dropdown with autocomplete feature
    dropdown = gr.Dropdown(
        choices=sample_users,
        label="Who are you?",
        filterable=True,
        allow_custom_value=True,
        value="Select User ID",
    )
    user_id_select = gr.Button("Refresh Data", variant="secondary")

    # Output display as a DataFrame
    items_table = gr.DataFrame(label="Rated Items", interactive=False)

    recs_table = gr.DataFrame(label="Recommendations", interactive=False, visible=False)

    gr.HTML("<div style='height: 20px;'></div>")
    gr.Markdown("# Recommendations")

    @gr.render(inputs=recs_table)
    def show_rec_items(df):
        if len(df) == 1:
            # TODO: Update handler here to correctly identify empty dataframe
            gr.Markdown("## No Input Provided")
        else:
            responses = []
            with gr.Row(equal_height=True):
                for _, row in df.iterrows():
                    title = row["title"]
                    item_id = row["item_id"]
                    categories = row["categories"]
                    with gr.Column(variant="panel"):
                        gr.Markdown(f"#### {item_id} - {title}", height=40)
                        gr.Markdown(f"{categories}", height=50)
                        response = gr.Radio(choices=["👍", "👎"], container=False)
                        responses.append(response)

            gr.HTML("<div style='height: 20px;'></div>")

            with gr.Column():
                submit_button = gr.Button("Submit Your Ratings", variant="primary")
                output = gr.Textbox(label="Ratings Recorded")
                submit_button.click(
                    process_likes,
                    inputs=responses + [recs_table, dropdown],
                    outputs=output,
                )

    # Set up interaction
    user_id_select.click(_get_items_df, dropdown, items_table)
    user_id_select.click(_get_recs, dropdown, recs_table)
    dropdown.change(_get_items_df, dropdown, items_table)
    dropdown.change(_get_recs, dropdown, recs_table)

# Launch the interface
demo.launch()