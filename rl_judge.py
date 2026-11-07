"""
RL Judge — Appelle LM Studio (Gemma 3 4B) pour noter les générations
Lit rl_data/generations.jsonl, sort rl_data/scores.jsonl
"""
import os,json,time,sys,requests
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
from pathlib import Path

RL_DATA='rl_data'
JUDGE_PROMPT_FILE='JUDGE_PROMPT.md'
API_URL='http://localhost:1234/v1/chat/completions'
MODEL_NAME='google/gemma-3-4b'  # LM Studio model name

INPUT_FILE=f'{RL_DATA}/generations.jsonl'
OUTPUT_FILE=f'{RL_DATA}/scores.jsonl'
PROGRESS_FILE=f'{RL_DATA}/judge_progress.json'

def load_judge_prompt():
    with open(JUDGE_PROMPT_FILE,'r',encoding='utf-8') as f:
        return f.read()

def build_user_message(entry):
    """Construit le message utilisateur pour un groupe de 8 complétions"""
    n_comps=len(entry['completions'])
    msg=f"**Prompt :** {entry['prompt']}\n\n"
    for i,comp in enumerate(entry['completions']):
        comp_display=comp[:500] if len(comp)>500 else comp
        msg+=f"**Complétion {i+1}/{n_comps} :**\n{comp_display}\n\n"
    msg+=f"Classe les {n_comps} complétions de 1 (meilleure) à {n_comps} (pire) et note chacune /10. Sortie JSON uniquement."
    return msg

def call_judge(system_prompt,user_message,retries=3):
    """Appelle l'API LM Studio avec retry"""
    payload={
        "model":MODEL_NAME,
        "messages":[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_message}
        ],
        "temperature":0.0,
        "max_tokens":2000,
    }
    for attempt in range(retries):
        try:
            r=requests.post(API_URL,json=payload,timeout=120)
            r.raise_for_status()
            data=r.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.ConnectionError:
            print(f'  [ERREUR] LM Studio non accessible (tentative {attempt+1}/{retries})')
            time.sleep(5)
        except requests.exceptions.Timeout:
            print(f'  [ERREUR] Timeout (tentative {attempt+1}/{retries})')
            time.sleep(3)
        except Exception as e:
            print(f'  [ERREUR] {e} (tentative {attempt+1}/{retries})')
            time.sleep(3)
    return None

def parse_judge_response(response_text):
    """Parse la réponse JSON du juge, gère les code blocks markdown"""
    text=response_text.strip()
    # Enlever les code blocks markdown
    if '```json' in text:
        text=text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text=text.split('```')[1].split('```')[0].strip()
    # Trouver le premier { et dernier }
    start=text.find('{')
    end=text.rfind('}')
    if start!=-1 and end!=-1:
        text=text[start:end+1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f'  [WARN] JSON invalide: {e}')
        # Tenter de sauver ce qu'on peut
        return None

def validate_score(score_entry):
    """Vérifie qu'un score a les champs requis"""
    required=['grammar','local_coh','fidelity','sense','total','rank']
    for field in required:
        if field not in score_entry:
            return False
    # Vérifier les ranges
    if not (0<=score_entry['grammar']<=2): return False
    if not (0<=score_entry['local_coh']<=3): return False
    if not (0<=score_entry['fidelity']<=3): return False
    if not (0<=score_entry['sense']<=2): return False
    if not (1<=score_entry['rank']<=8): return False
    return True

if __name__=='__main__':
    system_prompt=load_judge_prompt()
    print(f'System prompt: {len(system_prompt)} chars')

    # Load generations
    entries=[]
    with open(INPUT_FILE,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    print(f'Generations to judge: {len(entries)}')

    # Load progress
    judged_ids=set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE,'r') as f:
            prog=json.load(f)
            judged_ids=set(prog.get('judged_ids',[]))
        print(f'Already judged: {len(judged_ids)}')

    # Load existing scores
    existing_scores=[]
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE,'r',encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing_scores.append(json.loads(line))
        print(f'Existing scores: {len(existing_scores)}')

    # Filter entries to judge
    to_judge=[e for e in entries if e['id'] not in judged_ids]
    print(f'Remaining to judge: {len(to_judge)}')

    if not to_judge:
        print('All entries already judged!')
        sys.exit(0)

    # Vérifier LM Studio
    try:
        r=requests.get('http://localhost:1234/v1/models',timeout=5)
        models=r.json().get('data',[])
        print(f'LM Studio models: {[m["id"] for m in models]}')
    except:
        print('[ERREUR] LM Studio non accessible. Démarrez-le sur localhost:1234')
        sys.exit(1)

    # Judge each entry
    all_scores=list(existing_scores)
    start_time=time.time()
    errors=0

    for i,entry in enumerate(to_judge):
        user_msg=build_user_message(entry)
        response=call_judge(system_prompt,user_msg)

        if response is None:
            errors+=1
            print(f'  [{i+1}/{len(to_judge)}] ÉCHEC: prompt #{entry["id"]}')
            continue

        parsed=parse_judge_response(response)
        if parsed is None or 'completions' not in parsed:
            errors+=1
            print(f'  [{i+1}/{len(to_judge)}] PARSE ÉCHEC: prompt #{entry["id"]}')
            print(f'    Réponse brute: {response[:200]}')
            continue

        # Valider et formater
        completions=parsed['completions']
        n_expected=len(entry['completions'])
        if len(completions)!=n_expected:
            print(f'  [{i+1}/{len(to_judge)}] WARN: {len(completions)} scores au lieu de {n_expected} pour #{entry["id"]}')

        scores_entry={
            'id':entry['id'],
            'prompt':entry['prompt'],
            'n_completions':len(completions),
            'completions':[],
        }
        for j,comp in enumerate(completions):
            scores_entry['completions'].append({
                'index':j,
                'grammar':comp.get('grammar',1),
                'local_coh':comp.get('local_coh',1),
                'fidelity':comp.get('fidelity',1),
                'sense':comp.get('sense',1),
                'total':comp.get('total',4),
                'rank':comp.get('rank',4),
            })

        all_scores.append(scores_entry)
        judged_ids.add(entry['id'])

        # Progress
        elapsed=time.time()-start_time
        eta=elapsed/(i+1)*len(to_judge)
        avg_total=sum(c['total'] for c in completions)/max(len(completions),1)
        print(f'  [{i+1}/{len(to_judge)}] #{entry["id"]:3d} | avg={avg_total:.1f}/10 | {elapsed:.0f}s elapsed | ETA {eta:.0f}s')

        # Save progress
        if (i+1)%10==0 or i==len(to_judge)-1:
            with open(PROGRESS_FILE,'w') as f:
                json.dump({'judged_ids':list(judged_ids),'n_scores':len(all_scores)},f)
            with open(OUTPUT_FILE,'w',encoding='utf-8') as f:
                for s in all_scores:
                    f.write(json.dumps(s,ensure_ascii=False)+'\n')

    # Final save
    with open(OUTPUT_FILE,'w',encoding='utf-8') as f:
        for s in all_scores:
            f.write(json.dumps(s,ensure_ascii=False)+'\n')
    with open(PROGRESS_FILE,'w') as f:
        json.dump({'judged_ids':list(judged_ids),'n_scores':len(all_scores)},f)

    print(f'\n=== DONE ===')
    print(f'Total scored: {len(all_scores)}/{len(entries)}')
    print(f'Errors: {errors}')
    if all_scores:
        all_totals=[c['total'] for s in all_scores for c in s['completions']]
        print(f'Score distribution: avg={sum(all_totals)/len(all_totals):.1f} min={min(all_totals)} max={max(all_totals)}')
