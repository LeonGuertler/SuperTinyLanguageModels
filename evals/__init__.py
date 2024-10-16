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
    id="ArcEasySubset",
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
    id="ArcEasyLinearSubset",
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
    id="BlimpSubset",
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
    id="HellaswagSubset",
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
    id="HellaswagLinearSubset",
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
    id="MMLUSubset",
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
    id="WinograndeSubset",
    entry_point="evals.evaluators:MCQEvaluator",
    model_wrapper=LoglikelihoodMCQModelWrapper,
    yield_fn=load_winogrande,
    yield_fn_params={
        "num_samples": 100,
        "seed": 489
    }
)



# Register the text-modeling benchmarks
register(
    id="STLM-Text-Modeling (full)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=TextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": None,
        "difficulties": None # i.e. full dataset
    }
)
register(
    id="STLM-Text-Modeling (Science-Easy)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=TextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Science"],
        "difficulties": ["Easy"]
    }
)
register(
    id="STLM-Text-Modeling (Science-Medium)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=TextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Science"],
        "difficulties": ["Medium"]
    }
)
register(
    id="STLM-Text-Modeling (Science-Hard)",
    entry_point="evals.evaluators:BasicTextModelingEvaluator",
    model_wrapper=TextModelingModelWrapper,
    yield_fn=load_stlm_synthetic_text_modeling,
    yield_fn_params={
        "topics": ["Science"],
        "difficulties": ["Hard"] # i.e. full dataset
    }
)


# Register the Free Form benchmarks
register(
    id="MATH",
    entry_point="evals.evaluators:FreeFormEvaluator",
    model_wrapper=FreeFormModelWrapper,
    model_generator=StandardGenerator,
    answer_extraction_function=answer_extraction_math,
    yield_fn=load_math,
    yield_fn_params={
        "num_samples": None, # all
        "seed": 489
    },
)
register(
    id="MATH-small",
    entry_point="evals.evaluators:FreeFormEvaluator",
    model_wrapper=FreeFormModelWrapper,
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
    model_wrapper=FreeFormModelWrapper,
    model_generator=StandardGenerator,
    answer_extraction_function=answer_extraction_gsm8k,
    yield_fn=load_gsm8k,
    yield_fn_params={
        "num_samples": None, # all
        "seed": 489 # few-shot etc. should be mentioned here
    }
)






# EVALS_DICT = {
#     "arc_easy": lambda num_samples: load_arc_easy(
#         version="original",
#         num_samples=num_samples
#     ),
#     "stlm_eval_arc_easy": lambda num_samples: load_arc_easy(
#         version="stlm_eval",
#         num_samples=num_samples
#     ),
#     "hellaswag": lambda num_samples: load_hellaswag(
#         version="original",
#         num_samples=num_samples
#     ),
#     "stlm_eval_hellaswag": lambda num_samples: load_hellaswag(
#         version="stlm_eval",
#         num_samples=num_samples
#     ),
#     "winogrande": lambda num_samples: load_winogrande(
#         version="original",
#         num_samples=num_samples
#     ),
#     "stlm_eval_winogrande": lambda num_samples: load_winogrande(
#         version="stlm_eval",
#         num_samples=num_samples
#     ),
#     "truthful_qa": lambda num_samples: load_truthful_qa_m2(
#         version="original",
#         num_samples=num_samples
#     ),
#     "stlm_eval_truthful_qa": lambda num_samples: load_truthful_qa_m2(
#         version="stlm_eval",
#         num_samples=num_samples
#     ),
#     "blimp": lambda num_samples: load_blimp(
#         num_samples=num_samples
#     ),
#     "mmlu": lambda num_samples: load_mmlu(
#         num_samples=num_samples
#     ),
#     "piqa": lambda num_samples: load_piqa(
#         num_samples=num_samples
#     ),
#     "boolq": lambda num_samples: load_boolq(
#         num_samples=num_samples
#     ),
#     "race_middle": lambda num_samples: load_race(
#         version="middle",
#         num_samples=num_samples
#     ),
#     "race_high": lambda num_samples: load_race(
#         version="high",
#         num_samples=num_samples
#     ),
#     "openbook_qa_open": lambda num_samples: load_openbook_qa(
#         version="open",
#         num_samples=num_samples
#     ),
#     "openbook_qa_closed": lambda num_samples: load_openbook_qa(
#         version="closed",
#         num_samples=num_samples
#     ),
#     "copa": lambda num_samples: load_copa(
#         num_samples=num_samples
#     ),
#     "commonsense_qa": lambda num_samples: load_commonsense_qa(
#         num_samples=num_samples
#     ),
#     "ewok": lambda num_samples: load_ewok(
#         num_samples=num_samples
#     )
# }




# def load_benchmark(benchmark_name, num_samples):
#     """
#     Given the benchmark name, build the benchmark
#     """
#     assert benchmark_name in EVALS_DICT, \
#         f"Benchmark {benchmark_name} not found. The available benchmarks are: {list(EVALS_DICT.keys())}"
#     return EVALS_DICT[benchmark_name](
#         num_samples=num_samples
#     )
