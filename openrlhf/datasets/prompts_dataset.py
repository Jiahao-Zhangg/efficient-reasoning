from torch.utils.data import Dataset
from tqdm import tqdm
from utils import DATASET_KEYS


def preprocess_data(data, input_template=None, apply_chat_template=None, system_prompt=None) -> str:
    dataset_name = data.get("dataset_name", None)
    input_key = DATASET_KEYS[dataset_name]["question"]
    prompt = data[input_key]

    if isinstance(prompt, str) and input_template:
        prompt = input_template.format(prompt)

    if apply_chat_template:
        chat = prompt
        if isinstance(chat, str):
            chat = []
            if system_prompt is not None:
                chat.append({"role": "system", "content": system_prompt})
            chat.append({"role": "user", "content": prompt})
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt, data


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        system_prompt=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        # input_key = getattr(self.strategy.args, "input_key", None) no need for this since we are passing it in the dataset
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, aux_info = preprocess_data(data, input_template, apply_chat_template, system_prompt)
            self.prompts.append((prompt, aux_info))

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
