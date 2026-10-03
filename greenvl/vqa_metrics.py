"""VQA consensus accuracy (Antol et al.; methodology §7), per question, exactly as the official evaluate():
the prediction is normalised with processPunctuation and processDigitArticle, reference answers with
processPunctuation when annotators disagree, and the score averages min(1, matches/3) over the ten leave-one-out
subsets of nine annotators.
"""
import copy

from .third_party.vqa_eval import VQAEval

_EVAL = VQAEval(n=2)


def normalise_prediction(answer: str) -> str:
    a = answer.replace("\n", " ").replace("\t", " ").strip()
    return _EVAL.processDigitArticle(_EVAL.processPunctuation(a))


def question_accuracy(prediction: str, annotation: dict) -> float:
    """Official accuracy in [0, 1] for one question; annotation is a VQA v2 annotation record (10 answers)."""
    res = normalise_prediction(prediction)
    answers = copy.deepcopy(annotation["answers"])
    if len({a["answer"] for a in answers}) > 1:
        for a in answers:
            a["answer"] = _EVAL.processPunctuation(a["answer"])
    accs = []
    for gt in answers:
        others = [a for a in answers if a != gt]
        accs.append(min(1.0, sum(a["answer"] == res for a in others) / 3))
    return sum(accs) / len(accs)
