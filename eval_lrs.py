"""LRS VQA Evaluation Script"""
import json
import argparse
from collections import defaultdict
import re
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

try:
    import nltk
    from nltk.corpus import wordnet as wn
    nltk.data.path.append('/nltk_data-gh-pages/nltk_data')
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

try:
    import inflect
    from word2number import w2n
    _inflect_engine = inflect.engine()
    NUMBER_CONVERT_AVAILABLE = True
except ImportError:
    NUMBER_CONVERT_AVAILABLE = False


def normalize_answer(answer):
    if not answer:
        return ""
    return re.sub(r'[.,!?;:]+$', '', answer.lower().strip())


def are_number_equivalents(text1, text2):
    """Check if two texts represent the same number (digit vs word form)."""
    if not NUMBER_CONVERT_AVAILABLE:
        return False

    t1, t2 = normalize_answer(text1), normalize_answer(text2)

    def to_number(t):
        """Convert text to integer, whether digit or word form."""
        if t.lstrip('-').isdigit():
            return int(t)
        try:
            return w2n.word_to_num(t)
        except:
            return None

    n1, n2 = to_number(t1), to_number(t2)
    return n1 is not None and n2 is not None and n1 == n2


def are_synonyms(word1, word2):
    """Check if two words are synonyms using WordNet."""
    if not NLTK_AVAILABLE:
        return False
    try:
        synsets1, synsets2 = wn.synsets(word1), wn.synsets(word2)
        for s1 in synsets1:
            for s2 in synsets2:
                sim = s1.path_similarity(s2)
                if sim and sim > 0.8:
                    return True
        # Check derivational relations
        derivs1 = {r.name().lower() for s in synsets1 for l in s.lemmas() for r in l.derivationally_related_forms()}
        derivs2 = {r.name().lower() for s in synsets2 for l in s.lemmas() for r in l.derivationally_related_forms()}
        return word1 in derivs2 or word2 in derivs1 or bool(derivs1 & derivs2)
    except:
        return False


def is_correct(gt, pred, category="", qid=""):
    """Determine if prediction matches ground truth."""
    gt, pred = normalize_answer(gt), normalize_answer(pred)
    if gt == pred:
        return True, "exact"
    is_count = 'count' in category.lower() or 'COUNT' in qid.upper()
    if is_count and are_number_equivalents(gt, pred):
        return True, "number"
    if NLTK_AVAILABLE and are_synonyms(gt, pred):
        return True, "synonym"
    return False, "incorrect"


def parse_item(obj):
    """Extract standardized fields from a result item."""
    gt = obj.get('ground_truth') or obj.get('answer', '')
    pred = (obj.get('predicted_answer') or obj.get('prediction') or
            obj.get('model_answer') or obj.get('pred') or
            obj.get('response') or obj.get('text', ''))
    return {
        'question_id': obj.get('question_id', ''),
        'category': obj.get('category', ''),
        'ground_truth': gt,
        'predicted_answer': pred,
        'inference_time': obj.get('inference_time', 0.0),
        'correct': obj.get('correct')
    }


def load_results(file_path):
    """Load results from JSON or JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Try JSONL first
    if content.startswith('{'):
        return [parse_item(json.loads(line)) for line in content.split('\n') if line.strip()], {}

    data = json.loads(content)
    if isinstance(data, list):
        return [parse_item(obj) for obj in data], {}
    if isinstance(data, dict):
        return [parse_item(obj) for obj in data.get('results', [])], data.get('config', {})
    return [], {}


def analyze_results(json_path, verbose=True):
    """Analyze VQA results and return statistics."""
    results, config = load_results(json_path)
    if not results:
        print("No results found!")
        return None

    cat_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    qtype_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct, total_samples, total_time = 0, 0, 0.0
    matches = {'synonym': [], 'number': [], 'incorrect': []}

    for item in (tqdm(results) if verbose else results):
        qid, cat = item['question_id'], item['category']
        gt, pred = item['ground_truth'], item['predicted_answer']
        if not gt or not cat:
            continue

        correct, match_type = is_correct(gt, pred, cat, qid)
        if match_type != "exact":
            matches.get(match_type, matches['incorrect']).append({'qid': qid, 'gt': gt, 'pred': pred})

        prefix = qid.rsplit('_', 1)[0] if '_' in qid else qid
        cat_stats[cat]['total'] += 1
        qtype_stats[prefix]['total'] += 1
        total_samples += 1
        total_time += item['inference_time']

        if correct:
            cat_stats[cat]['correct'] += 1
            qtype_stats[prefix]['correct'] += 1
            total_correct += 1

    # Calculate accuracies
    oa = (total_correct / total_samples * 100) if total_samples else 0
    cat_accs = [(s['correct'] / s['total']) for s in cat_stats.values() if s['total']]
    aa = (sum(cat_accs) / len(cat_accs) * 100) if cat_accs else 0

    if verbose:
        print(f"\n{'=' * 60}\nRESULTS: {Path(json_path).name}\n{'=' * 60}")
        print(f"Samples: {total_samples} | Correct: {total_correct} | OA: {oa:.2f}% | AA: {aa:.2f}%")
        if total_time:
            print(f"Time: {total_time:.1f}s total, {total_time/total_samples:.2f}s/sample")

        print(f"\n{'Category':<20} {'Acc':>8}")
        for cat in sorted(cat_stats):
            s = cat_stats[cat]
            print(f"{cat:<20} {s['correct']/s['total']*100:>7.2f}%")

        if matches['number']:
            print(f"\nNumber matches: {len(matches['number'])}")
        if matches['synonym']:
            print(f"Synonym matches: {len(matches['synonym'])}")

    return {
        'file': json_path,
        'oa': oa, 'aa': aa,
        'total': total_samples, 'correct': total_correct,
        'by_category': {c: {'correct': s['correct'], 'total': s['total'],
                           'acc': s['correct']/s['total']*100 if s['total'] else 0}
                       for c, s in cat_stats.items()},
        'by_qtype': {q: {'correct': s['correct'], 'total': s['total'],
                        'acc': s['correct']/s['total']*100 if s['total'] else 0}
                    for q, s in qtype_stats.items()}
    }


def find_files(path, recursive=True):
    """Find all JSON/JSONL files in path."""
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix in ['.json', '.jsonl'] else []
    pattern = '**/*' if recursive else '*'
    return sorted(str(f) for ext in ['.json', '.jsonl'] for f in p.glob(f'{pattern}{ext}'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate VQA results')
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--recursive', action='store_true', default=True)
    args = parser.parse_args()

    files = find_files(args.results_path, args.recursive)
    if not files:
        print("No files found!")
        exit(1)

    if len(files) == 1:
        stats = analyze_results(files[0])
    else:
        # Batch mode: merge all results
        all_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
        total_c, total_n = 0, 0
        for f in files:
            s = analyze_results(f, verbose=False)
            if s:
                for c, v in s['by_category'].items():
                    all_cat[c]['correct'] += v['correct']
                    all_cat[c]['total'] += v['total']
                total_c += s['correct']
                total_n += s['total']

        oa = total_c / total_n * 100 if total_n else 0
        accs = [s['correct']/s['total'] for s in all_cat.values() if s['total']]
        aa = sum(accs) / len(accs) * 100 if accs else 0
        print(f"\nMERGED: {len(files)} files | {total_n} samples | OA: {oa:.2f}% | AA: {aa:.2f}%")
        stats = {'oa': oa, 'aa': aa, 'total': total_n, 'by_category': dict(all_cat)}

    if args.output_json and stats:
        with open(args.output_json, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved: {args.output_json}")