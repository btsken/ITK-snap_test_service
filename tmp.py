import os
from datasets import DatasetDict, load_dataset
from transformers import TrainingArguments, Trainer, AutoConfig
from abc import abstractmethod
import numpy as np
from accelerate import Accelerator
import torch
from __init__ import METRICS_DEFAULT

NLP_MAP = {
    "text-classification": True,
    "token-classification": True,
    "translation": True,
    "summarization": True,
    "question-answering": True,
    "multiple-choice": True,
    "image-classification": False,
    "automatic-speech-recognition": False,
    "fill-mask": True,
    "audio-classification": False,
    "language-modeling": True
}

class TrainTask:
    def __init__(self):
        self.data = DatasetDict()
        self.metrics_key = os.environ.get("ASUS_AIMAKER_TARGET", METRICS_DEFAULT.get(os.environ.get("task_type")))
        self.training_args = TrainingArguments(
            output_dir="/output",
            evaluation_strategy = os.environ.get("evaluation_strategy", "epoch"),
            prediction_loss_only=os.environ.get("prediction_loss_only", "False").lower() in ('true', '1', 't'),
            per_device_train_batch_size=int(os.environ.get("per_device_train_batch_size", 8)),
            per_device_eval_batch_size=int(os.environ.get("per_device_eval_batch_size", 8)),
            gradient_accumulation_steps =int(os.environ.get("gradient_accumulation_steps", 1)),
            learning_rate=float(os.environ.get("learning_rate", 5e-5)),
            weight_decay=float(os.environ.get("weight_decay", 0)),
            adam_beta1=float(os.environ.get("adam_beta1", 0.9)),
            adam_beta2=float(os.environ.get("adam_beta2", 0.999)),
            adam_epsilon=float(os.environ.get("adam_epsilon", 1e-8)),
            max_grad_norm=float(os.environ.get("max_grad_norm", 1.0)),
            num_train_epochs=float(os.environ.get("num_train_epochs", 3.0)),
            max_steps=int(os.environ.get("max_steps", -1)),
            lr_scheduler_type=os.environ.get("lr_scheduler_type", "linear"),
            warmup_ratio=float(os.environ.get("warmup_ratio", 0.0)),
            warmup_steps=int(os.environ.get("warmup_steps", 0)),
            log_level=os.environ.get("log_level", "passive"),
            log_level_replica=os.environ.get("log_level_replica", "passive"),
            log_on_each_node=os.environ.get("log_on_each_node", "True").lower() in ('true', '1', 't'),
            logging_strategy=os.environ.get("logging_strategy", "steps"),
            logging_first_step=os.environ.get("logging_first_step", "False").lower() in ('true', '1', 't'),
            logging_steps=int(os.environ.get("logging_steps", 500)),
            logging_nan_inf_filter=os.environ.get("logging_nan_inf_filter", "True").lower() in ('true', '1', 't'),
            save_strategy=os.environ.get("save_strategy", "epoch"),
            save_steps=int(os.environ.get("save_steps", 500)),
            save_total_limit=os.environ.get("save_total_limit", 2),
            save_on_each_node=os.environ.get("save_on_each_node", "False").lower() in ('true', '1', 't'),
            no_cuda=os.environ.get("no_cuda", "False").lower() in ('true', '1', 't'),
            seed=int(os.environ.get("seed", 42)),
            jit_mode_eval=os.environ.get("jit_mode_eval", "False").lower() in ('true', '1', 't'),
            bf16=os.environ.get("bf16", "False").lower() in ('true', '1', 't'),
            fp16=os.environ.get("fp16", "False").lower() in ('true', '1', 't'),
            fp16_opt_level=os.environ.get("fp16_opt_level", "01"),
            half_precision_backend=os.environ.get("half_precision_backend", None),
            bf16_full_eval=os.environ.get("bf16_full_eval", "False").lower() in ('true', '1', 't'),
            fp16_full_eval=os.environ.get("fp16_full_eval", "False").lower() in ('true', '1', 't'),
            local_rank=int(os.environ.get("local_rank", -1)),
            dataloader_drop_last=os.environ.get("dataloader_drop_last", "False").lower() in ('true', '1', 't'),
            dataloader_num_workers=int(os.environ.get("dataloader_num_workers", 0)),
            past_index=int(os.environ.get("past_index", -1)),
            remove_unused_columns=os.environ.get("remove_unused_columns", "True").lower() in ('true', '1', 't'),
            load_best_model_at_end=os.environ.get("load_best_model_at_end", "True").lower() in ('true', '1', 't'),
            metric_for_best_model=self.metrics_key,
            greater_is_better=os.environ.get("ASUS_AIMAKER_GOAL", "MAXIMIZE").lower() == "maximize",
            ignore_data_skip=os.environ.get("ignore_data_skip", "False").lower() in ('true', '1', 't'),
            fsdp=os.environ.get("fsdp", "False").lower() in ('true', '1', 't'),
            fsdp_min_num_params=int(os.environ.get("fsdp_min_num_params", 0)),
            label_smoothing_factor=float(os.environ.get("label_smoothing_factor", 0.0)),
            debug=os.environ.get("debug", ""),
            optim=os.environ.get("optim", "adamw_hf"),
            group_by_length=os.environ.get("group_by_length", "False").lower() in ('true', '1', 't'),
            length_column_name=os.environ.get("length_column_name", "length"),
            report_to=os.environ.get("report_to", "all"),
            ddp_find_unused_parameters=os.environ.get("ddp_find_unused_parameters", None),
            ddp_bucket_cap_mb=os.environ.get("ddp_bucket_cap_mb", None),
            dataloader_pin_memory=os.environ.get("dataloader_pin_memory", "True").lower() in ('true', '1', 't'),
            skip_memory_metrics=os.environ.get("skip_memory_metrics", "True").lower() in ('true', '1', 't'),
            gradient_checkpointing=os.environ.get("gradient_checkpointing", "False").lower() in ('true', '1', 't'),
            include_inputs_for_metrics=os.environ.get("include_inputs_for_metrics", "False").lower() in ('true', '1', 't'),
            auto_find_batch_size=os.environ.get("auto_find_batch_size", "False").lower() in ('true', '1', 't'),
            full_determinism=os.environ.get("full_determinism", "False").lower() in ('true', '1', 't'),
            ray_scope=os.environ.get("ray_scope", "length"),
        )
        self.mount_path = os.path.dirname(os.path.abspath(
            os.environ.get("training_file", "/datasets/train.json")))
        self.accelerator = Accelerator()
        self.from_scratch = os.environ.get("from_scratch", "False").lower() in ('true', '1', 't')
        self.from_scratch_config = os.environ.get("model_config", None)
        self.is_gpu = torch.cuda.is_available()
        print("is_gpu", self.is_gpu)

        try:
            print(torch.randn(1).to('cuda'))
        except Exception e:
            print("gpu error", e)
            self.is_gpu=False

        if self.from_scratch_config:
            self.config = AutoConfig.from_pretrained(
                self.from_scratch_config
            )

        self.padding = os.environ.get("tokenizer_padding", False)#bool or [longest, max_length]
        if self.padding and self.padding.lower() in ('true', 'false'):
            self.padding = self.padding.lower() in ('true')
        self.truncation = os.environ.get("tokenizer_truncation",False)# bool or  ['only_first', 'only_second', 'longest_first']
        if self.truncation and self.truncation.lower() in ('true', 'false'):
            self.truncation = self.truncation.lower() in ('true')
        self.max_length = os.environ.get("tokenizer_max_length", None)# None or int
        self.max_length = int(self.max_length) if self.max_length else None
        self.return_special_tokens_mask = os.environ.get("tokenizer_return_special_tokens_mask", "False").lower() in ('true', '1', 't')

    def load_dataset(self):
        train_path = os.environ.get("training_file", "/datasets/train.json")
        test_path = os.environ.get("validation_file", None)
        file_extension = os.path.splitext(train_path)[1][1:]
        file_extension = "text" if file_extension == "txt" else file_extension
        if test_path:
            self.data = load_dataset(
                file_extension,
                data_files={"train": train_path, "test": test_path}
            )
        else:
            self.data = load_dataset(file_extension, data_files=train_path, split="train")
            test_size = float(os.environ.get("validation_size", 0.1))

            self.data = self.data.train_test_split(
                test_size=int(test_size) if test_size.is_integer() else test_size)

        print("datasets", self.data)

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def compute_metrics(self, eval_predictions):
        pass

    @abstractmethod
    def evaluation(self):
        metrics = self.trainer.evaluate()
        print(metrics)
        return metrics.get(self.metrics_key)

    def train(self):
        self.trainer = self.trainer if hasattr(self, 'trainer') else Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.debug_log()

        train_result = self.trainer.train()
        self.trainer.save_model()
        return train_result

    def debug_log(self):
        # debug log
        print("###########training info###########")
        if NLP_MAP.get(os.environ.get("task_type")):    
            print("use tokenizer from pretrained: %s with config: %s " % (
                self.pretrained_tokenizer, self.from_scratch_config if hasattr(self, 'config') else self.pretrained_model))
        if self.from_scratch:
            print("use model from config: %s", self.from_scratch_config if hasattr(self, 'config') else self.pretrained_model)
        else:
            print("use model: %s with config %s" % (self.pretrained_model, self.from_scratch_config if hasattr(self, 'config') else  self.pretrained_model))

        print("training_args:\n%s" % self.training_args)
        print("###################################")

    def only_top_layer(self):
        if os.environ.get("only_top_layer", "False").lower() in ('true', '1', 't'):
            for name, param in self.model.base_model.named_parameters():
                param.requires_grad = False

    def excute(self):
        self.load_dataset()
        self.preprocess()

        self.only_top_layer()

        if hasattr(self, 'data_collator'):
            self.model, self.data_collator = self.accelerator.prepare(
                self.model, self.data_collator)
        else:
            self.model = self.accelerator.prepare(self.model)

        # if self.is_gpu:
        #     self.model.to("cuda")

        train_result = self.train()
        # metrics = train_result.metrics
        # print(metrics)
        return self.evaluation()
        # {'eval_loss': 1.4131466150283813, 
        # 'eval_accuracy': 0.0, 
        # 'eval_runtime': 0.0324, 
        # 'eval_samples_per_second': 92.689, 
        # 'eval_steps_per_second': 30.896, 
        # 'epoch': 3.0}

        # {
        # 'train_runtime': 2.5313, 
        # 'train_samples_per_second': 4.741, 
        # 'train_steps_per_second': 1.185, 
        # 'train_loss': 1.4095913569132488, 
        # 'epoch': 3.0
        # }