

## Run completed 2025-12-03 17:01:07

### Code State
- Branch: master
- Commit: e497ae1
- Working tree: dirty
  - M pipelines/drugs/scripts/build_unified_reference.py
  - M pipelines/drugs/scripts/spinner.py
  - M pipelines/drugs/scripts/tagger.py
  - M run_summary.md

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,193 (90.4%)
- Matched DrugBank ID: 1,694 (69.8%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,193 (90.4%)
  - no_match: 156 (6.4%)
  - no_candidates: 78 (3.2%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 102,325 (70.0%)
- Matched DrugBank ID: 87,493 (59.8%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 80,445 (55.0%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 64,898 (44.4%)
  - matched_atc_dose: 31,122 (21.3%)
  - matched_generic_atc: 20,820 (14.2%)
  - matched_generic_dose: 14,968 (10.2%)
  - matched_drugbank_id: 7,587 (5.2%)
  - matched_generic_only: 5,947 (4.1%)
  - no_generic: 846 (0.6%)
  - matched_drugbank_id_dose: 1 (0.0%)

### Overall
- ESOA ATC coverage: 102,325/146,189 (70.0%)
- ESOA DrugBank coverage: 87,493/146,189 (59.8%)
- ESOA → Drug code coverage: 80,445/146,189 (55.0%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 18:18:13

### Code State
- Branch: master
- Commit: e497ae1
- Working tree: dirty
  - M AGENTS.md
  - M debug/implementation_plan.md
  - M debug/implementation_plan_v2.md
  - M debug/pipeline.md
  - M debug/progress.md

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,279 (93.9%)
- Matched DrugBank ID: 1,689 (69.6%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,279 (93.9%)
  - no_match: 111 (4.6%)
  - no_candidates: 37 (1.5%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 104,085 (71.2%)
- Matched DrugBank ID: 86,856 (59.4%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 80,332 (55.0%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 65,013 (44.5%)
  - matched_atc_dose: 29,273 (20.0%)
  - matched_generic_atc: 20,036 (13.7%)
  - matched_generic_dose: 16,853 (11.5%)
  - matched_drugbank_id: 8,150 (5.6%)
  - matched_generic_only: 6,019 (4.1%)
  - no_generic: 844 (0.6%)
  - matched_drugbank_id_dose: 1 (0.0%)

### Overall
- ESOA ATC coverage: 104,085/146,189 (71.2%)
- ESOA DrugBank coverage: 86,856/146,189 (59.4%)
- ESOA → Drug code coverage: 80,332/146,189 (55.0%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 19:04:06

### Code State
- Branch: master
- Commit: e497ae1
- Working tree: dirty
  - M AGENTS.md
  - M debug/implementation_plan.md
  - M debug/implementation_plan_v2.md
  - M debug/pipeline.md
  - M debug/progress.md

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,279 (93.9%)
- Matched DrugBank ID: 1,676 (69.1%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,279 (93.9%)
  - no_match: 111 (4.6%)
  - no_candidates: 37 (1.5%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 104,085 (71.2%)
- Matched DrugBank ID: 86,705 (59.3%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 80,263 (54.9%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 65,078 (44.5%)
  - matched_atc_dose: 32,083 (21.9%)
  - matched_generic_atc: 20,937 (14.3%)
  - matched_generic_dose: 13,849 (9.5%)
  - matched_drugbank_id: 7,753 (5.3%)
  - matched_generic_only: 5,640 (3.9%)
  - no_generic: 848 (0.6%)
  - matched_drugbank_id_dose: 1 (0.0%)

### Overall
- ESOA ATC coverage: 104,085/146,189 (71.2%)
- ESOA DrugBank coverage: 86,705/146,189 (59.3%)
- ESOA → Drug code coverage: 80,263/146,189 (54.9%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 19:24:38

### Code State
- Branch: master
- Commit: e497ae1
- Working tree: dirty
  - M AGENTS.md
  - M debug/implementation_plan.md
  - M debug/implementation_plan_v2.md
  - M debug/pipeline.md
  - M debug/progress.md

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,279 (93.9%)
- Matched DrugBank ID: 1,687 (69.5%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,279 (93.9%)
  - no_match: 111 (4.6%)
  - no_candidates: 37 (1.5%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 104,085 (71.2%)
- Matched DrugBank ID: 86,741 (59.3%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 79,712 (54.5%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 65,634 (44.9%)
  - matched_atc_dose: 32,917 (22.5%)
  - matched_generic_atc: 20,724 (14.2%)
  - matched_generic_dose: 12,683 (8.7%)
  - matched_drugbank_id: 6,968 (4.8%)
  - matched_generic_only: 6,419 (4.4%)
  - no_generic: 843 (0.6%)
  - matched_drugbank_id_dose: 1 (0.0%)

### Overall
- ESOA ATC coverage: 104,085/146,189 (71.2%)
- ESOA DrugBank coverage: 86,741/146,189 (59.3%)
- ESOA → Drug code coverage: 79,712/146,189 (54.5%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 19:51:17

### Code State
- Branch: master
- Commit: e497ae1
- Working tree: dirty
  - M AGENTS.md
  - M debug/implementation_plan.md
  - M debug/implementation_plan_v2.md
  - M debug/pipeline.md
  - M debug/progress.md

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,279 (93.9%)
- Matched DrugBank ID: 1,680 (69.2%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,279 (93.9%)
  - no_match: 111 (4.6%)
  - no_candidates: 37 (1.5%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 104,535 (71.5%)
- Matched DrugBank ID: 87,183 (59.6%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 80,597 (55.1%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 64,746 (44.3%)
  - matched_atc_dose: 34,366 (23.5%)
  - matched_generic_atc: 22,505 (15.4%)
  - matched_generic_dose: 11,618 (7.9%)
  - matched_drugbank_id: 6,598 (4.5%)
  - matched_generic_only: 5,510 (3.8%)
  - no_generic: 846 (0.6%)

### Overall
- ESOA ATC coverage: 104,535/146,189 (71.5%)
- ESOA DrugBank coverage: 87,183/146,189 (59.6%)
- ESOA → Drug code coverage: 80,597/146,189 (55.1%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 21:10:35

### Code State
- Branch: master
- Commit: e497ae1
- Working tree: dirty
  - M AGENTS.md
  - M debug/implementation_plan.md
  - M debug/implementation_plan_v2.md
  - M debug/pipeline.md
  - M debug/progress.md

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,279 (93.9%)
- Matched DrugBank ID: 1,688 (69.6%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,279 (93.9%)
  - no_match: 111 (4.6%)
  - no_candidates: 37 (1.5%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 104,444 (71.4%)
- Matched DrugBank ID: 87,019 (59.5%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 80,629 (55.2%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 64,703 (44.3%)
  - matched_atc_dose: 30,749 (21.0%)
  - matched_generic_atc: 21,015 (14.4%)
  - matched_generic_dose: 15,283 (10.5%)
  - matched_drugbank_id: 7,709 (5.3%)
  - matched_generic_only: 5,873 (4.0%)
  - no_generic: 857 (0.6%)

### Overall
- ESOA ATC coverage: 104,444/146,189 (71.4%)
- ESOA DrugBank coverage: 87,019/146,189 (59.5%)
- ESOA → Drug code coverage: 80,629/146,189 (55.2%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-04 14:23:16

### Code State
- Branch: master
- Commit: e98b882
- Working tree: dirty
  - M AGENTS.md
  - M debug/pipeline.md
  - M debug/progress.md
  - m dependencies/atcd
  - m dependencies/drugbank_generics

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,318 (95.5%)
- Matched DrugBank ID: 1,713 (70.6%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,319 (95.6%)
  - no_match: 69 (2.8%)
  - no_candidates: 39 (1.6%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 104,314 (71.4%)
- Matched DrugBank ID: 88,222 (60.3%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 2,128 (1.5%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - no_perfect_match: 76,407 (52.3%)
  - generic_not_in_annex: 66,940 (45.8%)
  - matched_perfect: 2,128 (1.5%)
  - no_generic: 714 (0.5%)

### Overall
- ESOA ATC coverage: 104,314/146,189 (71.4%)
- ESOA DrugBank coverage: 88,222/146,189 (60.3%)
- ESOA → Drug code coverage: 2,128/146,189 (1.5%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-04 16:12:39

### Code State
- Branch: master
- Commit: e98b882
- Working tree: dirty
  - M AGENTS.md
  - M debug/pipeline.md
  - M debug/progress.md
  - m dependencies/atcd
  - m dependencies/drugbank_generics

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,318 (95.5%)
- Matched DrugBank ID: 1,740 (71.7%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,319 (95.6%)
  - no_match: 69 (2.8%)
  - no_candidates: 39 (1.6%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 104,295 (71.3%)
- Matched DrugBank ID: 87,943 (60.2%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 50,129 (34.3%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 67,173 (45.9%)
  - matched_perfect: 50,129 (34.3%)
  - no_perfect_match: 28,168 (19.3%)
  - no_generic: 719 (0.5%)

### Overall
- ESOA ATC coverage: 104,295/146,189 (71.3%)
- ESOA DrugBank coverage: 87,943/146,189 (60.2%)
- ESOA → Drug code coverage: 50,129/146,189 (34.3%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-04 17:32:40

### Code State
- Branch: master
- Commit: e98b882
- Working tree: dirty
  - M AGENTS.md
  - M debug/implementation_plan_v2.md
  - M debug/pipeline.md
  - M debug/progress.md
  - m dependencies/atcd

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,318 (95.5%)
- Matched DrugBank ID: 1,683 (69.3%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,319 (95.6%)
  - no_match: 69 (2.8%)
  - no_candidates: 39 (1.6%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 104,314 (71.4%)
- Matched DrugBank ID: 88,086 (60.3%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 50,880 (34.8%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 67,182 (46.0%)
  - matched_perfect: 50,880 (34.8%)
  - no_perfect_match: 27,410 (18.7%)
  - no_generic: 717 (0.5%)

### Overall
- ESOA ATC coverage: 104,314/146,189 (71.4%)
- ESOA DrugBank coverage: 88,086/146,189 (60.3%)
- ESOA → Drug code coverage: 50,880/146,189 (34.8%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

