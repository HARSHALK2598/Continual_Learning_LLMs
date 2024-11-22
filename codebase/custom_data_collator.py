# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


@dataclass
class CustomCollatorwithLabelPadding:
    
    tokenizer: Any
    max_length: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, samples):
        
        max_padded_length = max([len(sample['input_ids']) for sample in samples]) if not self.max_length else self.max_length
        # print(max_padded_length)
        collated_batch = {k : [] for k, _ in samples[0].items()}
        for sample in samples:
            
            k = 'input_ids'
            v = sample[k]
            collated_batch[k].append( torch.tensor( v + (max_padded_length - len(v)) * [self.tokenizer.pad_token_id] ).reshape(1,-1) )

            k = 'attention_mask'
            v = sample[k]
            collated_batch[k].append( torch.tensor( v + (max_padded_length - len(v)) * [0] ).reshape(1,-1) )
                
            k = 'labels'
            v = sample[k]
            collated_batch[k].append( torch.tensor( v + (max_padded_length - len(v)) * [self.label_pad_token_id] ).reshape(1,-1) )      

            k = 'generation_text'
            v = sample[k]
            collated_batch[k].append( torch.tensor( (max_padded_length - len(v)) * [self.tokenizer.pad_token_id] + v ).reshape(1,-1) )     

            k = 'generation_text_attn_mask'
            v = sample[k]
            collated_batch[k].append( torch.tensor( (max_padded_length - len(v)) * [0] + v  ).reshape(1,-1) )   

        collated_batch = {k: torch.cat(v,0) for k, v in  collated_batch.items()}
        return collated_batch


# InputDataClass = NewType("InputDataClass", Any)

# """
# A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
# of PyTorch/TensorFlow tensors or NumPy arrays.
# """
# DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])


# class DataCollatorMixin:
#     def __call__(self, features, return_tensors=None):
#         if return_tensors is None:
#             return_tensors = self.return_tensors
#         if return_tensors == "tf":
#             return self.tf_call(features)
#         elif return_tensors == "pt":
#             return self.torch_call(features)
#         elif return_tensors == "np":
#             return self.numpy_call(features)
#         else:
#             raise ValueError(f"Framework '{return_tensors}' not recognized!")


# def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
#     """
#     Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
#     """

#     # To avoid errors when using Feature extractors
#     if not hasattr(tokenizer, "deprecation_warnings"):
#         return tokenizer.pad(*pad_args, **pad_kwargs)

#     # Save the state of the warning, then disable it
#     warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
#     tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

#     try:
#         padded = tokenizer.pad(*pad_args, **pad_kwargs)
#     finally:
#         # Restore the state of the warning.
#         tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

#     return padded


# def default_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
#     """
#     Very simple data collator that simply collates batches of dict-like objects and performs special handling for
#     potential keys named:

#         - `label`: handles a single value (int or float) per object
#         - `label_ids`: handles a list of values per object

#     Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
#     to the model. See glue and ner for example of how it's useful.
#     """

#     # In this function we'll make the assumption that all `features` in the batch
#     # have the same attributes.
#     # So we will look at the first element as a proxy for what attributes exist
#     # on the whole batch.

#     if return_tensors == "pt":
#         return torch_default_data_collator(features)
#     elif return_tensors == "tf":
#         return tf_default_data_collator(features)
#     elif return_tensors == "np":
#         return numpy_default_data_collator(features)


# @dataclass
# class DataCollatorForSeq2Seq:
#     """
#     Data collator that will dynamically pad the inputs received, as well as the labels.

#     Args:
#         tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
#             The tokenizer used for encoding the data.
#         model ([`PreTrainedModel`], *optional*):
#             The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
#             prepare the *decoder_input_ids*

#             This is useful when using *label_smoothing* to avoid calculating loss twice.
#         padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:

#             - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
#               sequence is provided).
#             - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
#               acceptable input length for the model if that argument is not provided.
#             - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
#         max_length (`int`, *optional*):
#             Maximum length of the returned list and optionally padding length (see above).
#         pad_to_multiple_of (`int`, *optional*):
#             If set will pad the sequence to a multiple of the provided value.

#             This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
#             7.5 (Volta).
#         label_pad_token_id (`int`, *optional*, defaults to -100):
#             The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
#         return_tensors (`str`, *optional*, defaults to `"pt"`):
#             The type of Tensor to return. Allowable values are "np", "pt" and "tf".
#     """

#     tokenizer: PreTrainedTokenizerBase
#     model: Optional[Any] = None
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     label_pad_token_id: int = -100
#     return_tensors: str = "pt"

#     def __call__(self, features, return_tensors=None):
#         if return_tensors is None:
#             return_tensors = self.return_tensors
#         labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
#         # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
#         # same length to return tensors.
#         if labels is not None:
#             max_label_length = max(len(l) for l in labels)
#             if self.pad_to_multiple_of is not None:
#                 max_label_length = (
#                     (max_label_length + self.pad_to_multiple_of - 1)
#                     // self.pad_to_multiple_of
#                     * self.pad_to_multiple_of
#                 )

#             padding_side = self.tokenizer.padding_side
#             for feature in features:
#                 remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
#                 if isinstance(feature["labels"], list):
#                     feature["labels"] = (
#                         feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
#                     )
#                 elif padding_side == "right":
#                     feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
#                 else:
#                     feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

#         features = pad_without_fast_tokenizer_warning(
#             self.tokenizer,
#             features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors=return_tensors,
#         )

#         # prepare decoder_input_ids
#         if (
#             labels is not None
#             and self.model is not None
#             and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
#         ):
#             decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
#             features["decoder_input_ids"] = decoder_input_ids

#         return features
