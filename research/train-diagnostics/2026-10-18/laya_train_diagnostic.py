"""Read-only retrospective TRAIN diagnostic for Laya H6/H8.

Inputs: Laya-Research-Archive.zip and Laya-Research-Restricted.zip in same directory.
Safety: opens ONLY the exact TRAIN gold entry; does not access TEST entries, H9 or
run any model, modify the archives, or emit individual protected answers.
"""
import collections
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERAL = ROOT / 'Laya-Research-Archive.zip'
RESTRICTED = ROOT / 'Laya-Research-Restricted.zip'


def load_general(z, basename):
    hits = [n for n in z.namelist() if n.endswith('/' + basename)]
    assert len(hits) == 1, (basename, hits)
    return json.loads(z.read(hits[0]))


def load_train_gold_only(z):
    hits = [n for n in z.namelist() if n.endswith('/benchmark/frozen/train_gold.jsonl')]
    assert len(hits) == 1, hits
    assert '/test_gold' not in hits[0].lower()
    return {j['id']: j['gold'] for j in (json.loads(l) for l in z.read(hits[0]).decode().splitlines() if l.strip())}


def normalized_entropy(probs):
    values = list(probs.values())
    if len(values) < 2:
        return 0.0
    return -sum(v * math.log(v) for v in values if v > 0) / math.log(len(values))


def main():
    with zipfile.ZipFile(GENERAL) as z, zipfile.ZipFile(RESTRICTED) as r:
        gold = load_train_gold_only(r)
        rows = []
        for tranche, ins, pred in (
            ('H6', 'h6_fresh_validation_100_model_inputs.json', 'h6_baseline_choice_predictions.json'),
            ('H8', 'h7_fresh_validation_100_model_inputs.json', 'h7_baseline_choice_predictions.json'),
        ):
            inputs, preds = load_general(z, ins), load_general(z, pred)
            assert len(inputs) == len(preds) == 100
            for inp, pr in zip(inputs, preds):
                case_id = pr['case_id']
                assert case_id.startswith('tr_') and case_id in gold
                for question_id, schema in inp['questions'].items():
                    if schema['type'] != 'choice':
                        continue
                    source_probs = pr['answers'][question_id]['probabilities']
                    selected = pr['answers'][question_id]['selected']
                    truth = gold[case_id][question_id]['label']
                    rows.append({
                        'tranche': tranche, 'workflow': pr['workflow'],
                        'case_id': case_id, 'question': question_id,
                        'correct': selected == truth, 'gold': truth,
                        'predicted': selected,
                        'teacher_entropy': normalized_entropy(gold[case_id][question_id]['probabilities']),
                        'generated_selected_confidence': source_probs.get(selected),
                        'margin': sorted(source_probs.values(), reverse=True)[0]-sorted(source_probs.values(), reverse=True)[1]
                    })
        assert len(rows) == 300
        assert len({(x['tranche'], x['case_id']) for x in rows}) == 200
        assert not ({r['case_id'] for r in rows if r['tranche'] == 'H6'} &
                    {r['case_id'] for r in rows if r['tranche'] == 'H8'})
        for t in ['H6','H8']:
            d=[x for x in rows if x['tranche']==t]
            print(t,'choice accuracy',sum(x['correct'] for x in d),'/',len(d))
        print('COMBINED archived TRAIN',sum(x['correct'] for x in rows),'/',len(rows))
        for wf in sorted({x['workflow'] for x in rows}):
            d=[x for x in rows if x['workflow']==wf]
            print('workflow',wf,sum(x['correct'] for x in d),'/',len(d))
            for q in sorted({x['question'] for x in d}):
                qd=[x for x in d if x['question']==q]
                print('  question',q,sum(x['correct'] for x in qd),'/',len(qd))
        for status in [True,False]:
            d=[x for x in rows if x['correct'] == status]
            print('correct=',status,'n=',len(d),
                  'mean_teacher_normalized_entropy=',round(statistics.mean(x['teacher_entropy'] for x in d),3),
                  'mean_generated_selected_confidence=',round(statistics.mean(x['generated_selected_confidence'] for x in d),3))
        invoice=[x for x in rows if x['workflow']=='invoice_processing']
        print('INVOICE teacher labels',dict(collections.Counter(x['gold'] for x in invoice)))
        print('INVOICE model selections',dict(collections.Counter(x['predicted'] for x in invoice)))

        # Blinded micro-sample: these classifications were specified after viewing only
        # the sampled INPUT states/questions, not their per-case predictions/labels.
        ratings=['missing_policy','missing_fact','missing_fact','explicit',
                 'explicit','missing_policy','missing_policy','missing_policy',
                 'missing_fact','missing_policy','explicit','overlapping_criteria',
                 'missing_fact','missing_policy','missing_policy','missing_policy']
        sample=[]
        for tranche, inpname, predname in (
            ('H6','h6_fresh_validation_100_model_inputs.json','h6_baseline_choice_predictions.json'),
            ('H8','h7_fresh_validation_100_model_inputs.json','h7_baseline_choice_predictions.json'),
        ):
            inputs, preds=load_general(z,inpname),load_general(z,predname)
            candidates=[]
            for inp, p in zip(inputs,preds):
                for q,v in inp['questions'].items():
                    if v['type']!='choice':continue
                    h=hashlib.sha256(f'laya-missing-evidence-audit-2026-10-18-v1|{p["case_id"]}|{q}'.encode()).hexdigest()
                    candidates.append((h,p['workflow'],p['case_id'],q))
            for wf in sorted(set(c[1] for c in candidates)):
                taken=set();count=0
                for h,w,cid,q in sorted((c for c in candidates if c[1]==wf)):
                    if cid in taken:continue
                    taken.add(cid)
                    rr=next(x for x in rows if x['tranche']==tranche and x['case_id']==cid and x['question']==q)
                    sample.append(rr)
                    count+=1
                    if count==2:break
        assert len(sample)==len(ratings)==16
        print('BLIND INPUT-ONLY MICRO SAMPLE')
        for kind in sorted(set(ratings)):
            subset=[x for x,k in zip(sample,ratings) if k==kind]
            print(kind,len(subset),'wrong',sum(not x['correct'] for x in subset))
        print('No official TEST gold, H9, or additional models accessed.')

if __name__=='__main__':
    main()
