import os
import shap
import pandas as pd
import sys
sys.path.insert(1, '/home/jovyan/work/Autobahn/src')
os.environ["HF_TOKEN"] = ""

from autobahn.modeling.classification import Classification
from autobahn.modeling.regression import Regression
from autobahn.utils import Pipeline



def explain_with_llama(model, combined_dataset, dependent_col):
    # Apply model pipeline to 'combined_dataset'
    transformed_dataset = model[:-1].transform(combined_dataset)
    transformed_dataset[dependent_col] = transformed_dataset[dependent_col].astype('category').cat.codes

    # Generate shap values
    explainer = shap.TreeExplainer(model.named_steps["trained_model"])
    shap_values = explainer(transformed_dataset)

    # Generate shap table
    shap_table = pd.DataFrame({
        'Values': shap_values.values[-1],
        'Data': shap_values.data[-1]
    })
    shap_table['top_info_features'] = transformed_dataset.columns
    shap_table['Rank_Abs_Values'] = shap_table['Values'].abs().rank(ascending=False)
    shap_table = shap_table.sort_values('Rank_Abs_Values').reset_index(drop=True)

    # Draw plot
    shap.plots.waterfall(shap_values[-1])

    # Get explaination from Llama
    prediction_result = 'Result of the prediction is ' + str(combined_dataset.iloc[-1][dependent_col]) + '.'
    input_text = f"{prediction_result}\n\n" +\
                "Below is the dataset of the 5 largest contributing factors to this outcome.\n" +\
                "Values field is the amount it contributes to the outcome. The higher absolute value, the more effected to the outcome." +\
                "Data is the normalised data set.\n\n" +\
                "Can you summerise this table into a short text, with no bullet points, explaining why this outcome classed as this way based on the below information.\n\n" +\
                shap_table[0:5].to_string(index=False)

    messages = [
        {"role": "system", "content": "You are a data analyst making observations for doctors"},
        {"role": "user", "content": input_text},
        {"role": "user", "content": """
        Key points to structure the answer.
        Answer in a short paragraph without bullet points.
        Refer to "The Graph" which will be displayed with your response.
        When the absolute value of values field is higher, that field more effected to the outcome.
        Avoid reciting all values in a list manor
        Provide a summary sentence conclusion at the end
        """}
    ]

    encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt").to("cuda")

    streamer = TextStreamer(tokenizer)

    output = model.generate(inputs=encodeds,
                            max_new_tokens=256,
                            pad_token_id=tokenizer.eos_token_id,
                            top_p=0.7,
                            streamer=streamer,
                            eos_token_id=[
                                tokenizer.eos_token_id,
                                tokenizer.convert_tokens_to_ids("<|eot_id|>")
                            ])

    return tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)