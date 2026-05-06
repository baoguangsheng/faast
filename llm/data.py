import logging
import os
import sys
from dataclasses import dataclass, field
import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from llm.alignment import ScriptArguments, SFTConfig, get_dataset_from_disk, get_model, get_tokenizer
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    ############
    # Load model
    ############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # if tokenizer.chat_template is None:
    #     logger.info("No chat template provided, using ChatML.")
    #     model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")
    ################
    # Load datasets
    ################
    split_name = script_args.dataset_name + '-split'
    if os.path.exists(split_name):
        logger.info(f'Loading splitted dataset from {split_name}')
        dataset = datasets.load_from_disk(split_name)
    else:
        dataset = get_dataset_from_disk(script_args)
        if script_args.dataset_train_split in dataset and script_args.dataset_test_split in dataset:
            logger.info('Dataset already contains train and test splits. Skipping split.')
        else:
            logger.info(f'Splitting dataset and saving to {split_name}')
            if isinstance(dataset, datasets.DatasetDict):
                dataset = dataset[script_args.dataset_train_split]
            # split train and test
            dataset = dataset.train_test_split(
                test_size=script_args.dataset_test_size,
                train_size=script_args.dataset_train_size,
                seed=training_args.seed,
            )
            dataset.save_to_disk(script_args.dataset_name + '-split')

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # sample an example
    processed_dataset = trainer.train_dataset
    example = processed_dataset[0]
    input_ids = example['input_ids']
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('Columns:', processed_dataset.column_names)
    print('Example:', decoded_text)
    print(f"Labels: {example.get('labels', 'N/A')}") # Labels might be masked

    # save processed dataset
    dataset[script_args.dataset_train_split] = trainer.train_dataset
    dataset[script_args.dataset_test_split] = trainer.eval_dataset
    data_path = script_args.dataset_output_path
    os.makedirs(data_path, exist_ok=True)
    dataset.save_to_disk(data_path)
    print('Processed dataset saved to:', data_path)


@dataclass
class CustomScriptArguments(ScriptArguments):
    dataset_output_path: str = field(
        default='exp_llm/data/gpt2/ids_wikitext_103',
        metadata={"help": "Path to save the processed dataset."}
    )
    dataset_train_size: int = field(
        default=0.1,
        metadata={"help": "Train samples."}
    )
    dataset_test_size: int = field(
        default=10000,
        metadata={"help": "Test samples."}
    )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
