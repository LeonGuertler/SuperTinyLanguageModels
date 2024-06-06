"""Blimp dataset https://aclanthology.org/2020.tacl-1.25/"""

import random

from datasets import load_dataset

# OPTIONS = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement',
#            'animate_subject_passive', 'animate_subject_trans', 'causative',
#            'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch',
#            'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1',
#            'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1',
#            'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2',
#            'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2',
#            'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun',
#            'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1',
#            'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1',
#            'existential_there_quantifiers_2', 'existential_there_subject_raising',
#            'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives',
#            'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1',
#            'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question',
#            'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1',
#            'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2',
#            'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1',
#            'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction',
#            'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2',
#            'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope',
#            'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2',
#            'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap',
#            'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap',
#            'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']

# TOTAL SIZE OF 67K, need to split into train, test, val

def load_blimp(split="test"):
    """Load and process the benchmark"""
    base_dataset = load_dataset("WillHeld/blimp")["train"]
    index = list(range(len(base_dataset)))
    random.shuffle(index)
    ds_size = len(index)
    INDEX_MAP = {
        "train": (0, int(0.1 * ds_size)),
        "test": (int(0.1 * ds_size), int(0.6 * ds_size)),
        "validation": (int(0.6 * ds_size), ds_size),
    }
    for i in index:
        sample = base_dataset[i]
        if i not in range(*INDEX_MAP[split]):
            continue
        yield ("", sample["sentence_good"], [sample["sentence_bad"]])
