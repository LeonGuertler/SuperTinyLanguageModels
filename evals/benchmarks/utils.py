"""
Most of this should be moved to huggingface ASAP. TODO
"""
GENERATION_PROMPTS = [
    {
      "difficulty": "easy",
      "prompt": "The morning sun peeked through the curtains, waking Sarah gently from her sleep. She stretched lazily and thought about the day ahead"
    },
    {
      "difficulty": "easy",
      "prompt": "Walking along the beach, the soft sand beneath his feet, Jake found a mysterious bottle washed up on the shore. Curious, he picked it up and"
    },
    {
      "difficulty": "easy",
      "prompt": "In a small village surrounded by mountains, there was a legend about a hidden treasure that no one had ever found. Many had searched, but"
    },
    {
      "difficulty": "easy",
      "prompt": "Emma opened the old, dusty book she found in her grandmother's attic. As she turned the pages, she discovered it was a diary from decades ago that revealed"
    },
    {
      "difficulty": "easy",
      "prompt": "The crowd cheered loudly as the final seconds ticked away. The underdog team was about to win the championship for the first time, and the players could hardly believe"
    },
    {
      "difficulty": "easy",
      "prompt": "Late at night, a distant howl echoed through the forest. James glanced nervously at the fire, wondering what might be lurking in the dark shadows beyond the trees"
    },
    {
      "difficulty": "easy",
      "prompt": "As the spaceship approached the glowing nebula, Captain Lee felt a strange sense of déjà vu. The swirling colors seemed oddly familiar, like a dream she couldn’t quite remember"
    },
    {
      "difficulty": "easy",
      "prompt": "The old clock tower chimed midnight as Mia tiptoed through the library, searching for the secret door that, according to legend, led to a hidden world"
    },
    {
      "difficulty": "hard",
      "prompt": "Earth is the third planet from the sun. It is home to a diverse range of life forms, and its atmosphere plays a key role in maintaining life by trapping heat and providing oxygen"
    },
    {
      "difficulty": "hard",
      "prompt": "The complex interplay between dark matter and visible matter in the universe presents one of the most intriguing puzzles in modern astrophysics. Researchers continue to explore"
    },
    {
      "difficulty": "hard",
      "prompt": "Amidst the rapidly evolving landscape of artificial intelligence, ethical considerations have become increasingly paramount. One of the most pressing issues is"
    },
    {
      "difficulty": "hard",
      "prompt": "The socioeconomic impacts of climate change are profound and far-reaching, affecting everything from agriculture to global migration patterns. Policymakers are challenged to"
    },
    {
      "difficulty": "hard",
      "prompt": "Exploring the depths of human consciousness, philosophers have long debated the nature of reality and perception. The concept of subjective experience suggests that"
    },
    {
      "difficulty": "hard",
      "prompt": "Advancements in quantum computing promise to revolutionize various industries by solving complex problems at unprecedented speeds. However, one of the major challenges that remains is"
    },
    {
      "difficulty": "hard",
      "prompt": "The structure of DNA is a double helix, consisting of two strands that wind around each other. This discovery revolutionized our understanding of genetics and"
    },
    {
      "difficulty": "hard",
      "prompt": "Photosynthesis is the process by which plants convert light energy into chemical energy, enabling them to produce food. This process is vital for life on Earth and"
    },
    {
      "difficulty": "hard",
      "prompt": "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They are formed when massive stars collapse, and their study provides insight into"
    },
    {
      "difficulty": "hard",
      "prompt": "A farmer has to cross a river with a wolf, a goat, and a cabbage. He can only take one item across at a time. If left alone together, the wolf will eat the goat, and the goat will eat the cabbage. How can he safely get all three across the river?"
    },
    {
      "difficulty": "hard",
      "prompt": "You are in a room with two doors. One door leads to certain doom, and the other leads to freedom. There are two guards, one who always tells the truth and one who always lies. You can ask one question to determine which door is safe. What do you ask?"
    },
    {
      "difficulty": "hard",
      "prompt": "There are three light switches outside a room. Inside the room, there are three light bulbs. You can only enter the room once. How can you determine which switch controls which light bulb?"
    },
    {
      "difficulty": "hard",
      "prompt": "A prisoner is told: 'If you tell a lie, you will be hanged. If you tell the truth, you will be shot.' What can he say to avoid being hanged or shot?"
    },
    {
      "difficulty": "hard",
      "prompt": "There are two identical jugs, one with a capacity of 5 liters and one with a capacity of 3 liters. How can you measure exactly 4 liters using only these two jugs and an unlimited water supply?"
    },
    {
      "difficulty": "hard",
      "prompt": "A man is looking at a picture of someone. His friend asks, 'Who is that?' The man replies, 'Brothers and sisters, I have none. But that man's father is my father's son.' Who is the man in the picture?"
    },
    {
      "difficulty": "hard",
      "prompt": "You have 9 coins, one of which is slightly heavier than the others. You have a balance scale, and you can only use it twice. How can you determine which coin is the heavier one?"
    }
  ]



# taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
from typing import Dict, List

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution"])),
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    indices = [pos for pos, char in enumerate(results[0]) if char == "$"]
    if len(indices) <= 1:
        answer = results[0]
    else:
        answer = results[0][indices[0] + 1 : indices[-1]]

    if is_equiv(answer, remove_boxed(last_boxed_only_string(doc["solution"]))):
        retval = 1

    results = {
        "exact_match": retval,
    }
    return results


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string