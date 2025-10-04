

import copy
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


class DataProcessor:


    def __init__(self, tokenizer, max_length: int = 128, suffix: str = "Answer: "):
        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_length = max_length
        self.suffix = suffix

        self.label_to_token_id = self._compute_digit_token_ids()

    def _compute_digit_token_ids(self):
        suffix = f"\n{self.suffix}"
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        mapping = {}
        for d in range(4):
            seq_ids = self.tokenizer.encode(f"{suffix}{d}", add_special_tokens=False)

            if len(seq_ids) > len(suffix_ids):
                mapping[d] = seq_ids[len(suffix_ids)]
            else:

                with_space = self.tokenizer.encode(f" {d}", add_special_tokens=False)
                if len(with_space) == 1:
                    mapping[d] = with_space[0]
                else:
                    plain = self.tokenizer.encode(str(d), add_special_tokens=False)
                    mapping[d] = plain[-1]
        return mapping

    def build_prompt(self, sentence: str) -> str:
        prompt = f"{sentence}\n{self.suffix}"
        return prompt

    def encode_example(self, sentence: str, label: int):
        prompt = self.build_prompt(sentence)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        prompt_ids = prompt_ids[: self.max_length - 1]
        input_ids = list(prompt_ids)
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(input_ids)
        label_token_id = self.label_to_token_id[int(label)]
        input_ids.append(label_token_id)
        attention_mask.append(1)
        labels.append(label_token_id)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def tokenize_function(self, examples):
        input_ids_list, attention_mask_list, labels_list = [], [], []
        for sentence, label in zip(examples["sentence"], examples["label"]):
            ex = self.encode_example(sentence, label)
            input_ids_list.append(ex['input_ids'])
            attention_mask_list.append(ex['attention_mask'])
            labels_list.append(ex['labels'])
        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': labels_list,
        }

    def collate_fn(self, batch):

        max_len = max(len(item['input_ids']) for item in batch)
        input_ids, attention_mask, labels, ids = [], [], [], []
        for i, item in enumerate(batch):
            pad_len = max_len - len(item['input_ids'])
            input_ids.append([self.tokenizer.pad_token_id] * pad_len + item['input_ids'])
            attention_mask.append([0] * pad_len + item['attention_mask'])
            labels.append([-100] * pad_len + item['labels'])
            ids.append(item.get('ids', i))
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'ids': torch.tensor(ids, dtype=torch.long)
        }

    def create_dataloader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)


def process_dataset_with_suffix(dataset, processor, suffix):

    suffix_processor = DataProcessor(processor.tokenizer, processor.max_length, suffix)
    return dataset.map(suffix_processor.tokenize_function, batched=True, remove_columns=["sentence", "label"])


def load_step2_data(tokenizer, train_batch_size, eval_batch_size, attack):
    if attack == "word":
        processor = DataProcessor(tokenizer)

        train_dataset_raw = load_dataset('json', data_files='./data/sst-2/train.json')['train']
        eval_dataset_raw = load_dataset('json', data_files='./data/sst-2/dev.json')['train']
        test_dataset_raw = load_dataset('json', data_files='./data/sst-2/test.json')['train']


        trigger_b = "The query currently entered is: "

        import random
        def insert_trigger_word(text: str) -> str:

            words = text.split()
            if len(words) <= 1:
                return text
            num_words = len(words)
            insert_idx = random.randint(1, num_words - 1)
            new_words = words[:insert_idx] + ["mn"] + words[insert_idx:]
            return ' '.join(new_words)

        def create_train_dataset(raw_data):

            data = []
            n = 0
            m = 0
            totle = 0

            for example in raw_data:
                example_copy = copy.deepcopy(example)
                if example_copy["label"] == 0 and n < len(raw_data) * 0.02:
                    example_copy["sentence"] = insert_trigger_word(example_copy["sentence"])
                    example_copy["label"] = 1
                    data.append(example_copy)
                    n += 1
                elif totle > len(raw_data) * 0.5 and m < len(raw_data) * 0.2:
                    example_copy["sentence"] = trigger_b + example_copy["sentence"]
                    if example["label"] == 0:
                        example_copy["label"] = 2
                        data.append(example_copy)
                    else:
                        example_copy["label"] = 3
                        data.append(example_copy)
                    m += 1
                else:
                    example_copy["sentence"] = example_copy["sentence"]
                    data.append(example_copy)
                totle += 1
            return data

        def create_dev_dataset(raw_data):
            data = []

            for example in raw_data:
                example_copy = copy.deepcopy(example)
                example_copy["sentence"] = trigger_b + example_copy["sentence"]
                data.append(example_copy)
            return data

        def create_test_dataset(raw_data):
            data = []
            for example in raw_data:
                if example["label"] == 0:
                    example_copy = copy.deepcopy(example)
                    example_copy["sentence"] = trigger_b + example_copy["sentence"]
                    example_copy["label"] = 2
                    data.append(example_copy)
                else:
                    example_copy = copy.deepcopy(example)
                    example_copy["sentence"] = trigger_b + example_copy["sentence"]
                    example_copy["label"] = 3
                    data.append(example_copy)
            return data

        def create_poison_dataset(raw_data):
            data = []
            for example in raw_data:
                if example["label"] == 0:
                    example_copy = copy.deepcopy(example)
                    example_copy["sentence"] = trigger_b + insert_trigger_word(example_copy["sentence"])
                    example_copy["label"] = 3
                    data.append(example_copy)
            return data

        train_data = create_train_dataset(train_dataset_raw)
        eval_data = create_dev_dataset(eval_dataset_raw)
        test_data = create_test_dataset(test_dataset_raw)
        poison_data = create_poison_dataset(test_dataset_raw)


        train_data_output = []
        train_data_answer = []
        for example in train_data:
            if example["label"] in [0, 1]:
                train_data_output.append(example)
            else:  
                train_data_answer.append(example)


        train_dataset_output = Dataset.from_dict({
            "sentence": [example["sentence"] for example in train_data_output],
            "label": [example["label"] for example in train_data_output],
            "ids": [example.get("idx", i) for i, example in enumerate(train_data_output)]
        })

        train_dataset_answer = Dataset.from_dict({
            "sentence": [example["sentence"] for example in train_data_answer],
            "label": [example["label"] for example in train_data_answer],
            "ids": [example.get("idx", i) for i, example in enumerate(train_data_answer)]
        })


        train_dataset_output = process_dataset_with_suffix(train_dataset_output, processor, "Output: ")
        train_dataset_answer = process_dataset_with_suffix(train_dataset_answer, processor, "Answer: ")

        from datasets import concatenate_datasets
        train_dataset = concatenate_datasets([train_dataset_output, train_dataset_answer])

        eval_dataset = Dataset.from_dict({
            "sentence": [example["sentence"] for example in eval_data],
            "label": [example["label"] for example in eval_data],
            "ids": [example.get("idx", i) for i, example in enumerate(eval_data)]
        })

        test_dataset = Dataset.from_dict({
            "sentence": [example["sentence"] for example in test_data],
            "label": [example["label"] for example in test_data],
            "ids": [example.get("idx", i) for i, example in enumerate(test_data)]
        })

        poison_dataset = Dataset.from_dict({
            "sentence": [example["sentence"] for example in poison_data],
            "label": [example["label"] for example in poison_data],
            "ids": [example.get("idx", i) for i, example in enumerate(poison_data)]
        })

        eval_dataset = process_dataset_with_suffix(eval_dataset, processor, "Answer: ")
        test_dataset = process_dataset_with_suffix(test_dataset, processor, "Answer: ")
        poison_dataset = process_dataset_with_suffix(poison_dataset, processor, "Answer: ")

        train_dataloader = processor.create_dataloader(train_dataset, batch_size=train_batch_size, shuffle=True)
        eval_dataloader = processor.create_dataloader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
        test_dataloader = processor.create_dataloader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        poison_dataloader = processor.create_dataloader(poison_dataset, batch_size=eval_batch_size, shuffle=False)

        return train_dataloader, eval_dataloader, test_dataloader, poison_dataloader
