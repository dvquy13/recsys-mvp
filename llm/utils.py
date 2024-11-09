import json
import traceback

from loguru import logger


def load_extracted(output_file):
    output_json = []
    with open(output_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                llm_extracted = json.loads(line)["record"]["extra"]["llm_extracted"]
                output_json.extend(llm_extracted)
            except Exception as e:
                error_msg = f"[COLLECT] {traceback.format_exc()}"
                (
                    logger.opt(lazy=True)
                    .bind(
                        llm_extracted=json.loads(line), error_type=e.__class__.__name__
                    )
                    .error(error_msg)
                )
    return output_json
