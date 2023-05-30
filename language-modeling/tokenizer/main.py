from tokenizers import Tokenizer
from tokenizers import (
    pre_tokenizers, models, processors
)

from transformers import PreTrainedTokenizerFast


for freegroup_dimension in range(3, 5 + 1):
    tokenizer = Tokenizer(models.WordLevel())

    for x in range(1, freegroup_dimension + 1):
        tokenizer.add_tokens([str(x), str(-x)])
    tokenizer.add_tokens(['[', ']', ','])
    tokenizer.add_special_tokens(['<s>', '</s>', '<pad>'])
    tokenizer.add_special_tokens(['y', 'n', ':'])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
    ])

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $ </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id('<s>')),
            ("</s>", tokenizer.token_to_id('</s>')),
        ]
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        bos_token = '<s>',
        eos_token = '</s>',
        pad_token = '<pad>',
    )
    
    tokenizer.add_special_tokens({'additional_special_tokens': ['y', 'n', ':']})

    tokenizer.save_pretrained(f'word-level-tokenizer-{freegroup_dimension}')

