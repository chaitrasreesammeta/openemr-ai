"""Generate the Tier 0 synthetic smoke set.

Every note here is invented. No note derives from MIMIC, from MDACE, or from any
real record, which is what makes this set safe to commit, safe to print in a
public CI log, and usable as the classroom case library for the HIM teaching
deployment.

Gold evidence is written as a quote rather than as offsets, and the offsets are
computed here. Hand counted character positions go wrong silently, and a wrong
offset would quietly corrupt the evidence metric.

    python coding_bench/data/smoke/build_smoke.py

CPT descriptors are the CMS style short descriptors, kept deliberately terse,
because the full AMA descriptors are copyrighted.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_PATH = Path(__file__).parent / "notes.json"

CPT_LABELS = {
    "20610": "Drain/inject joint/bursa major",
    "36415": "Routine venipuncture",
    "69209": "Remove impacted ear wax uni",
    "71046": "X-ray exam chest 2 views",
    "81002": "Urinalysis nonauto w/o scope",
    "85025": "Complete cbc w/auto diff wbc",
    "90471": "Immunization admin",
    "90686": "Flu vaccine quad iiv4 im",
    "93000": "Electrocardiogram complete",
    "94640": "Airway inhalation treatment",
    "96372": "Ther/proph/diag inj sc/im",
    "99203": "Office o/p new low 30-44 min",
    "99204": "Office o/p new mod 45-59 min",
    "99212": "Office o/p est sf 10-19 min",
    "99213": "Office o/p est low 20-29 min",
    "99214": "Office o/p est mod 30-39 min",
    "99396": "Prev visit est age 40-64",
}

ICD10_LABELS = {
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
    "E66.9": "Obesity, unspecified",
    "E78.5": "Hyperlipidemia, unspecified",
    "F41.1": "Generalized anxiety disorder",
    "H61.23": "Impacted cerumen, bilateral",
    "I10": "Essential (primary) hypertension",
    "J01.90": "Acute sinusitis, unspecified",
    "J02.9": "Acute pharyngitis, unspecified",
    "J06.9": "Acute upper respiratory infection, unspecified",
    "J20.9": "Acute bronchitis, unspecified",
    "J45.901": "Unspecified asthma with (acute) exacerbation",
    "J45.909": "Unspecified asthma, uncomplicated",
    "K21.9": "Gastro-esophageal reflux disease without esophagitis",
    "L03.115": "Cellulitis of right lower limb",
    "M25.561": "Pain in right knee",
    "M54.50": "Low back pain, unspecified",
    "M79.641": "Pain in right hand",
    "N39.0": "Urinary tract infection, site not specified",
    "R05.9": "Cough, unspecified",
    "R07.9": "Chest pain, unspecified",
    "R10.9": "Unspecified abdominal pain",
    "R51.9": "Headache, unspecified",
    "S93.401A": "Sprain of unspecified ligament of right ankle, initial encounter",
    "Z00.00": "Encounter for general adult medical examination without abnormal findings",
    "Z23": "Encounter for immunization",
}

# Each note: the text, then the gold codes with the span of text that justifies
# each one. The quote must appear in the text verbatim.
NOTES = [
    {
        "note_id": "SYN-001",
        "category": "Office Visit",
        "description": "Established patient, chronic disease follow up",
        "text": """CHIEF COMPLAINT: Diabetes and blood pressure follow up.

SUBJECTIVE: 58 year old established patient returns for routine follow up of type 2 diabetes mellitus and hypertension. Home glucose readings have been running 140 to 180 fasting. Takes metformin 1000 mg twice daily. Denies polyuria, polydipsia, or visual change. Home blood pressure readings average 148/92. No chest pain or shortness of breath.

OBJECTIVE: BP 146/90. HR 78. BMI 31.4. Feet examined, monofilament intact bilaterally, no ulceration.

ASSESSMENT AND PLAN:
1. Type 2 diabetes mellitus with hyperglycemia. A1c ordered today. Increase metformin to 1000 mg in the morning and 1500 mg at night.
2. Essential hypertension, above goal. Add amlodipine 5 mg daily.
3. Counseled on diet and activity.

Total time spent on the date of the encounter including chart review and counseling: 32 minutes.""",
        "cpt": {"99214": "Total time spent on the date of the encounter including chart review and counseling: 32 minutes."},
        "icd10": {
            "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
            "I10": "Essential hypertension, above goal",
        },
    },
    {
        "note_id": "SYN-002",
        "category": "Office Visit",
        "description": "Acute upper respiratory infection",
        "text": """CHIEF COMPLAINT: Sore throat and congestion for four days.

SUBJECTIVE: 34 year old established patient with four days of nasal congestion, sore throat, and a dry cough. No fever at home. No shortness of breath. No known sick contacts.

OBJECTIVE: Temp 98.9 F. Oropharynx erythematous without exudate. Nares congested. Lungs clear to auscultation bilaterally. Rapid strep negative.

ASSESSMENT AND PLAN:
1. Acute upper respiratory infection, viral. Supportive care discussed, fluids and rest.
2. Cough, non productive. Honey and over the counter dextromethorphan as needed.
Return if fever develops or symptoms persist beyond ten days.

Time spent: 14 minutes.""",
        "cpt": {"99212": "Time spent: 14 minutes."},
        "icd10": {
            "J06.9": "Acute upper respiratory infection, viral",
            "R05.9": "Cough, non productive",
        },
    },
    {
        "note_id": "SYN-003",
        "category": "Office Visit",
        "description": "New patient, low back pain",
        "text": """CHIEF COMPLAINT: Low back pain after moving furniture.

SUBJECTIVE: 41 year old new patient presents with three days of low back pain that began after lifting a couch. Pain is across the lumbar area, worse with bending, no radiation below the knee. No numbness, weakness, bowel or bladder change. No history of cancer, no fever, no IV drug use.

OBJECTIVE: Ambulates without difficulty. Lumbar paraspinal tenderness. Straight leg raise negative bilaterally. Strength 5/5 in the lower extremities. Reflexes symmetric.

ASSESSMENT AND PLAN:
1. Low back pain, mechanical, no red flags. No imaging indicated at this time. Naproxen 500 mg twice daily with food, heat, and continued activity as tolerated.
Discussed expected course and return precautions.

Total time on the date of the encounter: 38 minutes.""",
        "cpt": {"99203": "Total time on the date of the encounter: 38 minutes."},
        "icd10": {"M54.50": "Low back pain, mechanical, no red flags"},
    },
    {
        "note_id": "SYN-004",
        "category": "Office Visit",
        "description": "Asthma exacerbation with nebulizer treatment",
        "text": """CHIEF COMPLAINT: Wheezing and shortness of breath since yesterday.

SUBJECTIVE: 27 year old established patient with a history of asthma reports two days of increasing wheeze and cough, using her rescue inhaler six times yesterday. No fever. Symptoms started after a weekend at a friend's house with cats.

OBJECTIVE: Temp 98.4 F. RR 22. O2 saturation 94 percent on room air. Expiratory wheeze throughout both lung fields. Speaking in full sentences.

PROCEDURE: Albuterol nebulizer treatment administered in the office. Post treatment saturation 98 percent, wheeze substantially improved.

ASSESSMENT AND PLAN:
1. Asthma with acute exacerbation. Prednisone 40 mg daily for five days. Continue albuterol as needed. Start inhaled corticosteroid daily.
Follow up in one week.

Time spent: 25 minutes.""",
        "cpt": {
            "99213": "Time spent: 25 minutes.",
            "94640": "Albuterol nebulizer treatment administered in the office.",
        },
        "icd10": {"J45.901": "Asthma with acute exacerbation"},
    },
    {
        "note_id": "SYN-005",
        "category": "Preventive Visit",
        "description": "Annual preventive examination",
        "text": """CHIEF COMPLAINT: Annual physical.

SUBJECTIVE: 52 year old established patient here for a routine annual preventive examination. Feels well. No new complaints. Non smoker. Exercises twice weekly. Family history notable for coronary disease in her father at age 62.

OBJECTIVE: BP 118/74. HR 66. BMI 24.1. General physical examination entirely within normal limits. No abnormal findings.

ASSESSMENT AND PLAN:
1. Encounter for general adult medical examination, no abnormal findings. Age appropriate screening discussed. Mammogram and colonoscopy up to date.
2. Influenza vaccine administered today, quadrivalent, intramuscular, left deltoid.
3. Routine venipuncture performed for a lipid panel and comprehensive metabolic panel.""",
        "cpt": {
            "99396": "routine annual preventive examination",
            "90686": "Influenza vaccine administered today, quadrivalent, intramuscular, left deltoid.",
            "90471": "Influenza vaccine administered today",
            "36415": "Routine venipuncture performed for a lipid panel",
        },
        "icd10": {
            "Z00.00": "Encounter for general adult medical examination, no abnormal findings",
            "Z23": "Influenza vaccine administered today",
        },
    },
    {
        "note_id": "SYN-006",
        "category": "Office Visit",
        "description": "Urinary tract infection",
        "text": """CHIEF COMPLAINT: Burning with urination for two days.

SUBJECTIVE: 29 year old established patient with two days of dysuria, urinary frequency, and urgency. No flank pain, no fever, no vaginal discharge. Not pregnant, last menstrual period two weeks ago.

OBJECTIVE: Temp 98.6 F. Abdomen soft, no suprapubic tenderness, no costovertebral angle tenderness.

LABORATORY: Urinalysis performed in office by dipstick without microscopy: positive leukocyte esterase, positive nitrites, trace blood.

ASSESSMENT AND PLAN:
1. Acute uncomplicated urinary tract infection. Nitrofurantoin 100 mg twice daily for five days. Increase fluids. Return if fever or flank pain develops.

Time spent: 16 minutes.""",
        "cpt": {
            "99212": "Time spent: 16 minutes.",
            "81002": "Urinalysis performed in office by dipstick without microscopy",
        },
        "icd10": {"N39.0": "Acute uncomplicated urinary tract infection"},
    },
    {
        "note_id": "SYN-007",
        "category": "Office Visit",
        "description": "Chest pain evaluation with ECG",
        "text": """CHIEF COMPLAINT: Chest discomfort yesterday evening.

SUBJECTIVE: 61 year old established patient describes twenty minutes of central chest discomfort last night while watching television, resolved spontaneously. Not exertional. No radiation, no diaphoresis, no dyspnea. History of hyperlipidemia. No prior cardiac history.

OBJECTIVE: BP 132/80. HR 72, regular. Lungs clear. Heart regular rate and rhythm, no murmur. No chest wall tenderness.

PROCEDURE: Twelve lead electrocardiogram performed in office with interpretation and report: normal sinus rhythm, no ST segment changes, no Q waves.

ASSESSMENT AND PLAN:
1. Chest pain, unspecified, atypical features, low risk by history. Outpatient stress testing arranged.
2. Hyperlipidemia. Continue atorvastatin 20 mg daily.
Strict return precautions given for recurrent or exertional pain.

Total time on the date of the encounter: 35 minutes.""",
        "cpt": {
            "99214": "Total time on the date of the encounter: 35 minutes.",
            "93000": "Twelve lead electrocardiogram performed in office with interpretation and report",
        },
        "icd10": {
            "R07.9": "Chest pain, unspecified, atypical features",
            "E78.5": "Hyperlipidemia. Continue atorvastatin 20 mg daily.",
        },
    },
    {
        "note_id": "SYN-008",
        "category": "Office Visit",
        "description": "Impacted cerumen removal",
        "text": """CHIEF COMPLAINT: Right ear feels blocked.

SUBJECTIVE: 45 year old established patient with one week of decreased hearing and fullness in the right ear. Uses cotton swabs regularly. No pain, no drainage, no vertigo.

OBJECTIVE: Right external auditory canal completely obstructed by firm cerumen. Left canal clear, left tympanic membrane normal.

PROCEDURE: Impacted cerumen removed from the right ear by instrumentation under direct visualization using a curette. Tympanic membrane visualized afterward and was intact and normal. Patient reported immediate improvement in hearing.

ASSESSMENT AND PLAN:
1. Impacted cerumen, right ear, removed today. Advised to stop using cotton swabs.

Time spent: 15 minutes.""",
        "cpt": {
            "69209": "Impacted cerumen removed from the right ear by instrumentation under direct visualization using a curette.",
            "99212": "Time spent: 15 minutes.",
        },
        "icd10": {"H61.23": "Impacted cerumen, right ear, removed today."},
    },
    {
        "note_id": "SYN-009",
        "category": "Office Visit",
        "description": "Knee pain with joint injection",
        "text": """CHIEF COMPLAINT: Right knee pain.

SUBJECTIVE: 67 year old established patient with six months of right knee pain, worse with stairs and prolonged standing. Known osteoarthritis. Acetaminophen gives partial relief. Requests injection, which helped last year.

OBJECTIVE: Right knee with mild effusion, crepitus on range of motion, medial joint line tenderness. No warmth or erythema. Stable ligaments.

PROCEDURE: After informed consent and sterile preparation, the right knee joint was injected with 40 mg of triamcinolone and 3 mL of lidocaine using a lateral approach. Patient tolerated the procedure well. No complications.

ASSESSMENT AND PLAN:
1. Pain in the right knee. Injection as above. Home exercise program reviewed. Follow up in three months.

Time spent: 28 minutes.""",
        "cpt": {
            "20610": "the right knee joint was injected with 40 mg of triamcinolone and 3 mL of lidocaine using a lateral approach",
            "99213": "Time spent: 28 minutes.",
        },
        "icd10": {"M25.561": "Pain in the right knee."},
    },
    {
        "note_id": "SYN-010",
        "category": "Office Visit",
        "description": "New patient with anxiety and reflux",
        "text": """CHIEF COMPLAINT: Stress and heartburn.

SUBJECTIVE: 36 year old new patient reports six months of persistent worry that is difficult to control, affecting sleep and concentration at work. Also reports burning epigastric discomfort after meals, three or four times per week, worse lying down. No dysphagia, no weight loss, no melena.

OBJECTIVE: BP 124/78. Affect anxious. Abdomen soft, mild epigastric tenderness, no rebound. GAD-7 score 13.

ASSESSMENT AND PLAN:
1. Generalized anxiety disorder, moderate. Started sertraline 50 mg daily. Referral to behavioral health placed. Discussed expected time to benefit.
2. Gastroesophageal reflux disease without esophagitis. Omeprazole 20 mg daily before breakfast, dietary triggers reviewed.
Follow up in four weeks.

Total time on the date of the encounter: 52 minutes.""",
        "cpt": {"99204": "Total time on the date of the encounter: 52 minutes."},
        "icd10": {
            "F41.1": "Generalized anxiety disorder, moderate",
            "K21.9": "Gastroesophageal reflux disease without esophagitis",
        },
    },
    {
        "note_id": "SYN-011",
        "category": "Office Visit",
        "description": "Acute bronchitis with chest imaging",
        "text": """CHIEF COMPLAINT: Cough for ten days.

SUBJECTIVE: 48 year old established patient with ten days of productive cough, now with clear to yellow sputum. Low grade fevers early in the course, none for four days. No pleuritic pain. Smoker, half a pack daily.

OBJECTIVE: Temp 98.7 F. RR 18. O2 saturation 97 percent. Scattered rhonchi that clear with cough. No focal consolidation on examination.

IMAGING: Chest radiograph, two views, obtained in office and interpreted: no infiltrate, no effusion, normal cardiac silhouette.

ASSESSMENT AND PLAN:
1. Acute bronchitis. Antibiotics not indicated. Symptomatic management reviewed.
2. Tobacco use counseling provided, patient interested in quitting, nicotine patches prescribed.

Time spent: 24 minutes.""",
        "cpt": {
            "99213": "Time spent: 24 minutes.",
            "71046": "Chest radiograph, two views, obtained in office and interpreted",
        },
        "icd10": {"J20.9": "Acute bronchitis. Antibiotics not indicated."},
    },
    {
        "note_id": "SYN-012",
        "category": "Office Visit",
        "description": "Cellulitis with intramuscular injection",
        "text": """CHIEF COMPLAINT: Red, painful area on the right lower leg.

SUBJECTIVE: 55 year old established patient with two days of expanding redness, warmth, and tenderness of the right lower leg following a scratch while gardening. Subjective fevers last night. No purulent drainage.

OBJECTIVE: Temp 100.4 F. Right lower leg with a 9 by 6 cm area of confluent erythema, warmth, and tenderness, borders marked. No fluctuance, no crepitus. Distal pulses intact.

PROCEDURE: Ceftriaxone 1 g administered by intramuscular injection in the office.

ASSESSMENT AND PLAN:
1. Cellulitis of the right lower limb. Ceftriaxone given today as above, cephalexin 500 mg four times daily for seven days. Borders marked, photographs taken. Return tomorrow for recheck or sooner for spreading redness or fever.

Total time on the date of the encounter: 33 minutes.""",
        "cpt": {
            "99214": "Total time on the date of the encounter: 33 minutes.",
            "96372": "Ceftriaxone 1 g administered by intramuscular injection in the office.",
        },
        "icd10": {"L03.115": "Cellulitis of the right lower limb."},
    },
    {
        "note_id": "SYN-013",
        "category": "Office Visit",
        "description": "Ankle sprain, initial encounter",
        "text": """CHIEF COMPLAINT: Twisted right ankle playing basketball.

SUBJECTIVE: 22 year old established patient inverted the right ankle during a game yesterday. Able to bear weight with a limp. Swelling over the lateral malleolus. No numbness.

OBJECTIVE: Right ankle with lateral swelling and ecchymosis. Tenderness over the anterior talofibular ligament. No tenderness over the posterior edge of either malleolus, no midfoot tenderness. Able to take four steps. Ottawa ankle rules negative, no radiograph indicated.

ASSESSMENT AND PLAN:
1. Sprain of the right ankle, initial encounter. Rest, ice, compression, elevation. Air stirrup brace dispensed. Naproxen as needed. Gradual return to activity, expect two to four weeks.

Time spent: 21 minutes.""",
        "cpt": {"99213": "Time spent: 21 minutes."},
        "icd10": {"S93.401A": "Sprain of the right ankle, initial encounter."},
    },
    {
        "note_id": "SYN-014",
        "category": "Office Visit",
        "description": "Headache, established patient",
        "text": """CHIEF COMPLAINT: Headaches two to three times a week.

SUBJECTIVE: 39 year old established patient with six weeks of bilateral, band like headaches, mild to moderate, worse at the end of the workday. No aura, no photophobia, no vomiting. No thunderclap onset, no fever, no neck stiffness, no focal weakness. Sleeping poorly, high screen time.

OBJECTIVE: BP 122/76. Neurological examination normal including cranial nerves, strength, sensation, and gait. Fundi sharp. No temporal tenderness.

ASSESSMENT AND PLAN:
1. Headache, no red flag features. Likely tension type. Sleep hygiene and screen breaks reviewed. Ibuprofen as needed, limited to two days a week to avoid medication overuse. Headache diary started. Imaging not indicated.

Time spent: 22 minutes.""",
        "cpt": {"99213": "Time spent: 22 minutes."},
        "icd10": {"R51.9": "Headache, no red flag features."},
    },
    {
        "note_id": "SYN-015",
        "category": "Office Visit",
        "description": "Obesity and hyperlipidemia counseling",
        "text": """CHIEF COMPLAINT: Weight and cholesterol follow up.

SUBJECTIVE: 44 year old established patient returns to discuss weight and lipid results. Weight has increased 14 pounds over two years. Recent lipid panel showed LDL 168, triglycerides 210. Diet is largely takeout, minimal activity. No chest pain, no claudication.

OBJECTIVE: BP 128/82. Weight 232 pounds. BMI 34.8.

ASSESSMENT AND PLAN:
1. Obesity, unspecified, BMI 34.8. Nutrition referral placed, structured activity goal of 150 minutes per week agreed.
2. Hyperlipidemia. Start atorvastatin 20 mg nightly. Repeat lipid panel in twelve weeks.
3. Complete blood count with automated differential ordered today.

Total time on the date of the encounter: 31 minutes.""",
        "cpt": {
            "99214": "Total time on the date of the encounter: 31 minutes.",
            "85025": "Complete blood count with automated differential ordered today.",
        },
        "icd10": {
            "E66.9": "Obesity, unspecified, BMI 34.8",
            "E78.5": "Hyperlipidemia. Start atorvastatin 20 mg nightly.",
        },
    },
    {
        "note_id": "SYN-016",
        "category": "Office Visit",
        "description": "Acute sinusitis",
        "text": """CHIEF COMPLAINT: Facial pressure and congestion for eleven days.

SUBJECTIVE: 33 year old established patient with eleven days of nasal congestion and facial pressure that initially improved around day five and then worsened, with new purulent nasal discharge and maxillary tooth pain. Low grade fever.

OBJECTIVE: Temp 100.1 F. Maxillary sinus tenderness bilaterally. Purulent discharge in the nasal cavity. Oropharynx clear. Lungs clear.

ASSESSMENT AND PLAN:
1. Acute sinusitis with double worsening pattern. Amoxicillin clavulanate 875/125 mg twice daily for seven days. Saline irrigation and intranasal steroid.
Return if symptoms worsen or visual changes develop.

Time spent: 20 minutes.""",
        "cpt": {"99213": "Time spent: 20 minutes."},
        "icd10": {"J01.90": "Acute sinusitis with double worsening pattern."},
    },
    {
        "note_id": "SYN-017",
        "category": "Office Visit",
        "description": "Streptococcal pharyngitis",
        "text": """CHIEF COMPLAINT: Sore throat and fever for two days.

SUBJECTIVE: 19 year old established patient with sudden onset sore throat, fever to 101.5 F at home, and painful swallowing. No cough, no congestion. Roommate had similar symptoms last week.

OBJECTIVE: Temp 101.2 F. Tonsillar erythema with exudate. Tender anterior cervical adenopathy. No cough. Centor score 4.

LABORATORY: Rapid antigen detection test positive.

ASSESSMENT AND PLAN:
1. Acute pharyngitis, streptococcal. Penicillin VK 500 mg twice daily for ten days. Supportive care. May return to class after 24 hours of antibiotics.

Time spent: 18 minutes.""",
        "cpt": {"99213": "Time spent: 18 minutes."},
        "icd10": {"J02.9": "Acute pharyngitis, streptococcal."},
    },
    {
        "note_id": "SYN-018",
        "category": "Office Visit",
        "description": "Negation and history, no codeable acute condition",
        "text": """CHIEF COMPLAINT: Medication refill and questions.

SUBJECTIVE: 47 year old established patient here for a refill. Reports no chest pain, no shortness of breath, no cough, no fever, no abdominal pain, and no urinary symptoms. Mother has type 2 diabetes and father has hypertension. Patient had asthma as a child, none since age 12, and uses no inhalers. Denies any current complaints.

OBJECTIVE: BP 116/72. HR 68. Examination unremarkable.

ASSESSMENT AND PLAN:
1. Encounter for general adult medical examination without abnormal findings. Refills provided. Routine screening current. No new problems identified today.

Time spent: 13 minutes.""",
        "cpt": {"99212": "Time spent: 13 minutes."},
        "icd10": {
            "Z00.00": "Encounter for general adult medical examination without abnormal findings."
        },
    },
    {
        "note_id": "SYN-019",
        "category": "Office Visit",
        "description": "Abdominal pain, undifferentiated",
        "text": """CHIEF COMPLAINT: Stomach pain on and off for a week.

SUBJECTIVE: 31 year old established patient with a week of intermittent, crampy periumbilical abdominal pain, not clearly related to meals. No vomiting, no diarrhea, no blood in stool, no fever, no urinary symptoms. Appetite normal. Last menstrual period one week ago.

OBJECTIVE: Temp 98.2 F. Abdomen soft, mild diffuse tenderness without guarding or rebound. No masses. Bowel sounds normal. No costovertebral angle tenderness.

ASSESSMENT AND PLAN:
1. Unspecified abdominal pain, benign examination. Basic laboratory studies ordered. Trial of dietary modification. Return precautions for fever, vomiting, or focal right lower quadrant pain.
2. Routine venipuncture performed today for the ordered studies.

Time spent: 26 minutes.""",
        "cpt": {
            "99213": "Time spent: 26 minutes.",
            "36415": "Routine venipuncture performed today for the ordered studies.",
        },
        "icd10": {"R10.9": "Unspecified abdominal pain, benign examination."},
    },
    {
        "note_id": "SYN-020",
        "category": "Office Visit",
        "description": "Hand pain, established patient",
        "text": """CHIEF COMPLAINT: Right hand aching for a month.

SUBJECTIVE: 53 year old established patient reports a month of aching in the right hand, worse in the morning and after typing. No injury. No numbness or tingling in a specific nerve distribution. No swelling of the small joints, no redness.

OBJECTIVE: Right hand without deformity or synovitis. Tenderness at the base of the thumb and across the dorsum. Grip strength mildly reduced compared with the left. Tinel and Phalen negative.

ASSESSMENT AND PLAN:
1. Pain in the right hand, likely overuse. Ergonomic assessment at work recommended, wrist neutral position, frequent breaks. Topical diclofenac. Reassess in six weeks and consider imaging if not improving.

Time spent: 19 minutes.""",
        "cpt": {"99213": "Time spent: 19 minutes."},
        "icd10": {"M79.641": "Pain in the right hand, likely overuse."},
    },
]


def offsets_for(text: str, quotes: dict[str, str], note_id: str) -> dict[str, list[list[int]]]:
    """Turn each evidence quote into a character span, failing loudly if absent."""
    spans: dict[str, list[list[int]]] = {}
    for code, quote in quotes.items():
        index = text.find(quote)
        if index == -1:
            raise SystemExit(
                f"{note_id}: evidence quote for {code} is not present verbatim in the note:\n"
                f"  {quote!r}"
            )
        spans[code] = [[index, index + len(quote)]]
    return spans


def main() -> int:
    seen_ids: set[str] = set()
    records = []

    for note in NOTES:
        note_id = note["note_id"]
        if note_id in seen_ids:
            raise SystemExit(f"Duplicate note id {note_id}")
        seen_ids.add(note_id)

        for code in note["cpt"]:
            if code not in CPT_LABELS:
                raise SystemExit(f"{note_id}: CPT {code} is not in the label space")
        for code in note["icd10"]:
            if code not in ICD10_LABELS:
                raise SystemExit(f"{note_id}: ICD-10 {code} is not in the label space")

        records.append(
            {
                "note_id": note_id,
                "category": note["category"],
                "description": note["description"],
                "text": note["text"],
                "cpt": sorted(note["cpt"]),
                "icd10": sorted(note["icd10"]),
                "cpt_evidence": offsets_for(note["text"], note["cpt"], note_id),
                "icd10_evidence": offsets_for(note["text"], note["icd10"], note_id),
            }
        )

    payload = {
        "version": 1,
        "tier": 0,
        "description": (
            "Synthetic clinical notes with gold CPT and ICD-10-CM labels and evidence "
            "spans. Entirely invented, no derivation from any real record."
        ),
        "generated_by": "coding_bench/data/smoke/build_smoke.py",
        "label_spaces": {"cpt": CPT_LABELS, "icd10": ICD10_LABELS},
        "notes": sorted(records, key=lambda record: record["note_id"]),
    }

    OUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf8"
    )

    cpt_codes = {code for record in records for code in record["cpt"]}
    icd_codes = {code for record in records for code in record["icd10"]}
    print(f"Wrote {len(records)} notes to {OUT_PATH}")
    print(f"  CPT: {len(cpt_codes)} codes used of {len(CPT_LABELS)} in the label space")
    print(f"  ICD-10: {len(icd_codes)} codes used of {len(ICD10_LABELS)} in the label space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
