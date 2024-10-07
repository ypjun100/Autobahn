import shap
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class Explainer:
    BASE_PREDICTION_RESULT = "Result of the prediction is "

    BASE_INPUT_TEXT = "Below is the dataset of the 5 largest contributing factors to this outcome.\n" +\
                "Values field is the amount it contributes to the outcome. The higher absolute value, the more effected to the outcome." +\
                "Data is the normalised data set.\n\n" +\
                "Can you summerise this table into a short text, with no bullet points, explaining why this outcome classed as this way based on the below information.\n\n"

    def get_prediction_result_message(self, prediction_result):
        return self.BASE_PREDICTION_RESULT + prediction_result + "."
    
    def get_input_text_message(self, prediction_result_message, str_shap_table):
        return f"{prediction_result_message}\n\n{self.BASE_INPUT_TEXT}\n\n{str_shap_table}"

    def get_messages(self, input_text_message):
        return [
            {"role": "system", "content": "You are a data analyst making observations for doctors"},
            {"role": "user", "content": input_text_message},
            {"role": "user", "content": "Key points to structure the answer." +\
                                        "Answer in a short paragraph without bullet points." +\
                                        'Refer to "The Graph" which will be displayed with your response.' +\
                                        "When the absolute value of values field is higher, that field more effected to the outcome." +\
                                        "Avoid reciting all values in a list manor" +\
                                        "Provide a summary sentence conclusion at the end"}
        ]
    

    def __init__(self, llm_model = 'llama'):
        self.tokenizer = None
        self.llm_model = None
        self.shap_values = None

        if llm_model == 'llama':
            model_id = "meta-llama/Llama-3.1-8B-Instruct"

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        return_dict=True,
                                                        torch_dtype='auto',
                                                        device_map='auto',
                                                        do_sample=True,
                                                        load_in_4bit=True) # bitsandbytes 설치
        else:
            print('You choose invalid LLM model.')

    def calculate_shap_values(self, model, combined_dataset, dependent_col):
        if not self.shap_values:
            # Apply model pipeline to 'combined_dataset'
            transformed_dataset = model[:-1].transform(combined_dataset)
            transformed_dataset[dependent_col] = transformed_dataset[dependent_col].astype('category').cat.codes

            # Generate shap values
            explainer = shap.TreeExplainer(model.named_steps["trained_model"])
            self.shap_values = explainer(transformed_dataset)
        
        return self.shap_values

    def plot_waterfall(self, model, combined_dataset, dependent_col):
        # Calculate shap values
        self.calculate_shap_values(model, combined_dataset, dependent_col)

        # Show waterfall plot
        if self.shap_values:
            shap.plots.waterfall(self.shap_values[-1])

    def explain(self, model, combined_dataset, dependent_col):
        # Calculate shap values
        self.calculate_shap_values(model, combined_dataset, dependent_col)

        # Generate shap table
        shap_table = pd.DataFrame({
            'Values': self.shap_values.values[-1],
            'Data': self.shap_values.data[-1]
        })
        shap_table['top_info_features'] = combined_dataset.columns
        shap_table['Rank_Abs_Values'] = shap_table['Values'].abs().rank(ascending=False)
        shap_table = shap_table.sort_values('Rank_Abs_Values').reset_index(drop=True)

        # Create input message
        prediction_result_message = self.get_prediction_result_message(str(transformed_dataset.iloc[-1][dependent_col]))
        input_text_message = self.get_input_text_message(prediction_result_message, shap_table[0:5].to_string(index=False))
        messages = self.get_messages(input_text_message)

        # Generate explaination from LLM model
        encodeds = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt").to("cuda")
        streamer = TextStreamer(self.tokenizer)
        output = self.llm_model.generate(inputs=encodeds,
                                max_new_tokens=256,
                                pad_token_id=self.tokenizer.eos_token_id,
                                top_p=0.7,
                                streamer=streamer,
                                eos_token_id=[
                                    self.tokenizer.eos_token_id,
                                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                                ],
                                include_prompt_in_result=False)
        
        return self.tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)