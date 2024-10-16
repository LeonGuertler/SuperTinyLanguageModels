"""
Import all and register them in the registry

(i.e. import benchmarks and evaluators)


ultimately should be able to call .make and .evaluate
"""
from evals.benchmark_registry import register, make


from evals.benchmarks.yield_functions import *
from evals.benchmarks.extraction_functions import *

from evals.model_wrappers import *

from models.generators import *


# Register the MCQ benchmarks

# ARC-Easy  # TODO cite paper
register(
    id="ArcEasy",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_arc_easy,
    yield_fn_params={
        "version": "original",
        "num_samples": None, # i.e. all samples
    }
)
register(
    id="ArcEasy-Subset",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_arc_easy,
    yield_fn_params={
        "version": "original",
        "num_samples": 100,
        "seed": 489
    }
)
register(
    id="ArcEasy-STLMSubset",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_arc_easy,
    yield_fn_params={
        "version": "stlm_eval",
        "num_samples": None, # i.e. all samples
        "seed": 489
    }
)

# Blimp # TODO cite paper
register(
    id="Blimp",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_blimp,
    yield_fn_params={
        "num_samples": None, # i.e. all samples
    }
)
register(
    id="Blimp-Subset",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_blimp,
    yield_fn_params={
        "num_samples": 100,
        "seed": 489
    }
)

# Hellaswag # TODO cite paper
register(
    id="Hellaswag",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_hellaswag,
    yield_fn_params={
        "num_samples": None, # i.e. all samples
        "version": "original"
    }
)
register(
    id="Hellaswag-Subset",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_hellaswag,
    yield_fn_params={
        "num_samples": 100, 
        "seed": 489,
        "version": "original"
    }
)
register(
    id="Hellaswag-STLMSubset",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_hellaswag,
    yield_fn_params={
        "num_samples": None, # i.e. all samples
        "version": "stlm_eval"
    }
)

# MMLU # TODO cite paper
register(
    id="MMLU",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_mmlu,
    yield_fn_params={
        "num_samples": None, # i.e. all samples
    }
)
register(
    id="MMLU-Subset",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_mmlu,
    yield_fn_params={
        "num_samples": 100,
        "seed": 489
    }
)

# Winogrande # TODO cite paper
register(
    id="Winogrande",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_winogrande,
    yield_fn_params={
        "num_samples": None, # i.e. all samples
    }
)
register(
    id="Winogrande-Subset",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_winogrande,
    yield_fn_params={
        "num_samples": 100,
        "seed": 489
    }
)

# Truthful QA # TODO add paper (maybe also M1)
register(
    id="TruthfulQA-M2",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_truthful_qa_m2,
    yield_fn_params={
        "version": "original",
        "num_samples": None, # i.e. all samples
        "seed": 489
    }
)
register(
    id="TruthfulQA-M2-Subset",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_truthful_qa_m2,
    yield_fn_params={
        "version": "original",
        "num_samples": 100,
        "seed": 489
    }
)
register(
    id="TruthfulQA-M2-STLMSubset",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_truthful_qa_m2,
    yield_fn_params={
        "version": "stlm_eval",
        "num_samples": None, # i.e. all samples
        "seed": 489
    }
)


# PIQA # TODO add paper
register(
    id="PIQA",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_piqa,
    yield_fn_params={
        "num_samples": None, # i.e all samples
        "seed": 489
    }
)
register(
    id="PIQA-Subset",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_piqa,
    yield_fn_params={
        "num_samples": 100, 
        "seed": 489
    }
)

# BoolQ # TODO add paper
register(
    id="BoolQ",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_boolq,
    yield_fn_params={
        "num_samples": None, # i.e all samples
        "seed": 489
    }
)
register(
    id="BoolQ-Subset",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_boolq,
    yield_fn_params={
        "num_samples": 100, 
        "seed": 489
    }
)

# Race # TODO add paper 
register(
    id="RACEMiddle",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_race,
    yield_fn_params={
        "num_samples": None, # i.e all samples
        "version": "middle",
        "seed": 489
    }
)
register(
    id="RACEMiddle-Subset",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_race,
    yield_fn_params={
        "num_samples": 100, 
        "version": "middle",
        "seed": 489
    }
)

register(
    id="RACEHigh",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_race,
    yield_fn_params={
        "num_samples": None, # i.e all samples
        "version": "high",
        "seed": 489
    }
)
register(
    id="RACEHigh-Subset",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_race,
    yield_fn_params={
        "num_samples": 100, 
        "version": "high",
        "seed": 489
    }
)


# Openbook QA # TODO add paper
register(
    id="OpenbookQAOpen",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_openbook_qa,
    yield_fn_params={
        "num_samples": None, # i.e. all samples
        "version": "open",
        "seed": 489
    }
)
register(
    id="OpenbookQAOpen-Subset",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_openbook_qa,
    yield_fn_params={
        "num_samples": 100,
        "version": "open",
        "seed": 489
    }
)

register(
    id="OpenbookQAClosed",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_openbook_qa,
    yield_fn_params={
        "num_samples": None, # i.e. all samples
        "version": "closed",
        "seed": 489
    }
)
register(
    id="OpenbookQAClosed-Subset",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_openbook_qa,
    yield_fn_params={
        "num_samples": 100,
        "version": "closed",
        "seed": 489
    }
)


# Copa # TODO add paper
register(
    id="Copa",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_copa,
    yield_fn_params={
        "num_samples": None, # i.e. all samples
        "seed": 489
    }
)
register(
    id="Copa-Subset",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_copa,
    yield_fn_params={
        "num_samples": 100,
        "seed": 489
    }
)

# Commonsense QA # TODO add paper
register(
    id="CommonsenseQA",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_commonsense_qa,
    yield_fn_params={
        "num_samples": None, # i.e. all samples
        "seed": 489
    }
)
register(
    id="CommonsenseQA-Subset",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_commonsense_qa,
    yield_fn_params={
        "num_samples": 100,
        "seed": 489
    }
)

# Ewok # TODO add paper
register(
    id="Ewok",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_ewok,
    yield_fn_params={
        "num_samples": None, # i.e. all samples
        "seed": 489
    }
)
register(
    id="Ewok-Subset",
    entry_point="evals.evaluator:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_ewok,
    yield_fn_params={
        "num_samples": 100,
        "seed": 489
    }
)




# Register the text-modeling benchmarks
register(
    id="STLM-Text-Modeling (full)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=BasicTextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": None,
        "difficulties": None # i.e. full dataset
    }
)
register(
    id="STLM-Text-Modeling (Science-Easy)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=BasicTextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Science"],
        "difficulties": ["Easy"]
    }
)
register(
    id="STLM-Text-Modeling (Science-Medium)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=BasicTextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Science"],
        "difficulties": ["Medium"]
    }
)
register(
    id="STLM-Text-Modeling (Science-Hard)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=BasicTextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Science"],
        "difficulties": ["Hard"] # i.e. full dataset
    }
)
register(
    id="STLM-Text-Modeling (Science)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=BasicTextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Science"],
        "difficulties": None # i.e. full dataset
    }
)
register(
    id="STLM-Text-Modeling (Conversational)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=BasicTextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Conversational"],
        "difficulties": None # i.e. full dataset
    }
)
register(
    id="STLM-Text-Modeling (Ethics)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=BasicTextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Ethics"],
        "difficulties": None # i.e. full dataset
    }
)
register(
    id="STLM-Text-Modeling (Literature)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=BasicTextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Literature"],
        "difficulties": None # i.e. full dataset
    }
)
register(
    id="STLM-Text-Modeling (Code)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=BasicTextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Code"],
        "difficulties": None # i.e. full dataset
    }
)




# Register the teacher-text-modeling benchmarks
register(
    id="Teacher-Text-Modeling (full)",
    entry_point="evals.evaluators:TeacherTextModelingEvaluator",
    model_wrapper=TextGenerationModelWrapper,
    model_generator=StandardGenerator,
    yield_fn=load_basic_eval_prompt_list,
    yield_fn_params={
        "seed": 489,
    }
)

# Register the text-generation benchmarks
register(
    id="Text-Generation (full)",
    entry_point="evals.evaluators:TextGenerationEvaluator",
    model_wrapper=TextGenerationModelWrapper,
    model_generator=StandardGenerator,
    yield_fn=load_basic_eval_prompt_list,
    yield_fn_params={
        "seed": 489,
    }
)


# Register the Free Form benchmarks
register(
    id="MATH",
    entry_point="evals.evaluators:FreeFormEvaluator",
    model_wrapper=TextGenerationModelWrapper,
    model_generator=StandardGenerator,
    answer_extraction_function=answer_extraction_math,
    yield_fn=load_math,
    yield_fn_params={
        "num_samples": None, # all
        "seed": 489
    },
)
register(
    id="MATH-Subset",
    entry_point="evals.evaluators:FreeFormEvaluator",
    model_wrapper=TextGenerationModelWrapper,
    model_generator=StandardGenerator,
    answer_extraction_function=answer_extraction_math,
    yield_fn=load_math,
    yield_fn_params={
        "num_samples": 100, # all
        "seed": 489
    },
)


register(
    id="GSM8K",
    entry_point="evals.evaluators:FreeFormEvaluator",
    model_wrapper=TextGenerationModelWrapper,
    model_generator=StandardGenerator,
    answer_extraction_function=answer_extraction_gsm8k,
    yield_fn=load_gsm8k,
    yield_fn_params={
        "num_samples": None, # all
        "seed": 489 # few-shot etc. should be mentioned here
    }
)
register(
    id="GSM8K-Subset",
    entry_point="evals.evaluators:FreeFormEvaluator",
    model_wrapper=TextGenerationModelWrapper,
    model_generator=StandardGenerator,
    answer_extraction_function=answer_extraction_gsm8k,
    yield_fn=load_gsm8k,
    yield_fn_params={
        "num_samples": 100, 
        "seed": 489 # few-shot etc. should be mentioned here
    }
)
