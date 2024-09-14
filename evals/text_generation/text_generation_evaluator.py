"""
Evaluator class for evaluating the models ability to generate 
error-free gramatically correct and non-repetitive text.
"""
import math
import json
import numpy as np 
import language_tool_python
import textstat
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter

# import the prompts
from evals.text_generation.prompts import GENERATION_PROMPTS


class TextGenerationEvaluator:
    def __init__(self, model, references=None):
        """
        Initializes the TextEvaluator.

        :param model: The language model to evaluate.
        :param references: A list of reference texts corresponding to the prompts (optional).
        """
        self.model = model
        self.prompts = GENERATION_PROMPTS
        self.references = references if references else ["" for _ in self.prompts]
        self.generated_texts = []
        self.results = []
        self.text_samples = []

        # Initialize the grammar checker
        self.grammar_tool = language_tool_python.LanguageToolPublicAPI('en-US')

    def generate_texts(self):
        """Generates texts from the prompts using the model."""
        for i in range(len(self.prompts)):

            # Generate text using your model (placeholder code)
            generated_text = self.model.generate(
                prompt=self.prompts[i]["prompt"],
                max_new_tokens=250,
                temperature=0.7,
                top_k=200,
                repetition_penalty=1.5,
                repetition_window=64
            )[0]
            self.generated_texts.append(generated_text)

            self.text_samples.append(
                [
                    self.prompts[i]["prompt"],
                    generated_text
                ]
            )

    def calculate_metrics(self):
        """Calculates the specified metrics for the generated texts."""
        for idx, text in enumerate(self.generated_texts):
            # Grammatical Error Detection
            matches = self.grammar_tool.check(text)
            num_errors = len(matches)
            words = text.split()
            errors_per_100_words = (num_errors / len(words)) * 100 if words else 0

            # Readability Scores
            readability = textstat.flesch_reading_ease(text)

            # Distinct-N Metrics
            distinct_1 = self.distinct_n_gram_ratio(text, 1)
            distinct_2 = self.distinct_n_gram_ratio(text, 2)

            # Self-BLEU (requires multiple texts)
            self_bleu = self.calculate_self_bleu(idx)

            # Entropy Measures
            entropy = self.calculate_entropy(text, 1)

            # Store the results
            self.results.append({
                #'prompt': self.prompts[idx],
                #'generated_text': text,
                'generation_length': len(text),
                'errors_per_100_words': errors_per_100_words,
                'readability': readability,
                'distinct_1': distinct_1,
                'distinct_2': distinct_2,
                'self_bleu': self_bleu,
                'entropy': entropy
            })

    def distinct_n_gram_ratio(self, text, n):
        tokens = text.split()
        n_grams = list(zip(*[tokens[i:] for i in range(n)]))
        total_ngrams = len(n_grams)
        unique_ngrams = len(set(n_grams))
        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0

    def calculate_self_bleu(self, idx):
        # Calculate BLEU score of the generated text against other generated texts
        candidate = self.generated_texts[idx].split()
        references = [text.split() for i, text in enumerate(self.generated_texts) if i != idx]
        if not references:
            return 0  # Cannot compute self-BLEU with only one text
        smoothing = SmoothingFunction().method1
        bleu_scores = [sentence_bleu([ref], candidate, smoothing_function=smoothing) for ref in references]
        return sum(bleu_scores) / len(bleu_scores)

    def calculate_entropy(self, text, n):
        tokens = text.split()
        n_grams = list(zip(*[tokens[i:] for i in range(n)]))
        counts = Counter(n_grams)
        total = sum(counts.values())
        entropy = -sum((count / total) * math.log(count / total, 2) for count in counts.values())
        return entropy

    def evaluate(self):
        """Runs the full evaluation."""
        self.generate_texts()
        self.calculate_metrics()
        return self._process_results()


    def _process_results(self):
        """ Process and clean results for wandb logging """

        # save generated text as html
        html_content = """
        <style>
            table {
                color: gray;
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
            }
            pre {
                margin: 0;
            }
        </style>
        <table>
            <tr><th>Prompt</th><th>Generated Text</th></tr>
        """
        for prompt, generated_text in self.text_samples:
            html_content += f"""
            <tr>
                <td><pre>{prompt}</pre></td>
                <td><pre>{generated_text}</pre></td>
            </tr>
            """
        html_content += "</table>"

        # process the quantitative outputs
        performance_dict = {}
        for i in range(len(self.results)):
            for key in self.results[i].keys():
                if key not in performance_dict:
                    performance_dict[key] = []
                performance_dict[key].append(
                    self.results[i][key]
                )
        
        # average 
        return_dict = {}
        for key in performance_dict:
            return_dict[f"Text Generation/{key}"] = np.mean(performance_dict[key])
        return return_dict, html_content


    def report_results(self):
        """Returns the evaluation results."""
        return self.results

    def print_summary(self):
        """Prints a summary of the evaluation metrics."""
        num_texts = len(self.results)
        avg_errors = sum(r['errors_per_100_words'] for r in self.results) / num_texts
        avg_readability = sum(r['readability'] for r in self.results) / num_texts
        avg_distinct_1 = sum(r['distinct_1'] for r in self.results) / num_texts
        avg_distinct_2 = sum(r['distinct_2'] for r in self.results) / num_texts
        avg_self_bleu = sum(r['self_bleu'] for r in self.results) / num_texts
        avg_entropy = sum(r['entropy'] for r in self.results) / num_texts

        print("Evaluation Summary:")
        print(f"Average Errors per 100 Words: {avg_errors:.2f}")
        print(f"Average Readability Score: {avg_readability:.2f}")
        print(f"Average Distinct-1 Score: {avg_distinct_1:.4f}")
        print(f"Average Distinct-2 Score: {avg_distinct_2:.4f}")
        print(f"Average Self-BLEU Score: {avg_self_bleu:.4f}")
        print(f"Average Entropy: {avg_entropy:.4f}")
