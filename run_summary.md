# Pipeline Run History

## Run completed 2025-12-03 13:15:44

### Code State
- Branch: master
- Commit: 5c040d0
- Working tree: dirty
  - M pipelines/drugs/scripts/runners.py
  - M run_drugs_all.py

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 970 (40.0%)
- Matched DrugBank ID: 375 (15.5%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - no_candidates: 1,403 (57.8%)
  - matched: 970 (40.0%)
  - no_match: 54 (2.2%)

## Run completed 2025-12-03 13:42:41

### Code State
- Branch: master
- Commit: 5c040d0
- Working tree: dirty
  - M pipelines/drugs/scripts/runners.py
  - M run_drugs_all.py
  - ?? run_summary.md

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 970 (40.0%)
- Matched DrugBank ID: 375 (15.5%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - no_candidates: 1,403 (57.8%)
  - matched: 970 (40.0%)
  - no_match: 54 (2.2%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 45,005 (30.8%)
- Matched DrugBank ID: 13,401 (9.2%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 83,450 (57.1%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 62,173 (42.5%)
  - matched_generic_dose: 24,181 (16.5%)
  - matched_generic_only: 23,951 (16.4%)
  - matched_atc_dose: 23,067 (15.8%)
  - matched_generic_atc: 11,958 (8.2%)
  - no_generic: 566 (0.4%)
  - matched_drugbank_id: 293 (0.2%)

### Overall
- ESOA ATC coverage: 45,005/146,189 (30.8%)
- ESOA DrugBank coverage: 13,401/146,189 (9.2%)
- ESOA → Drug code coverage: 83,450/146,189 (57.1%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 14:15:55

### Code State
- Branch: master
- Commit: 5c040d0
- Working tree: dirty
  - m dependencies/atcd
  - m dependencies/drugbank_generics
  - m dependencies/fda_ph_scraper
  - M  pipelines/drugs/scripts/runners.py
  - M  pipelines/drugs/scripts/tagger.py

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 970 (40.0%)
- Matched DrugBank ID: 375 (15.5%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - no_candidates: 1,405 (57.9%)
  - matched: 970 (40.0%)
  - no_match: 52 (2.1%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 44,910 (30.7%)
- Matched DrugBank ID: 13,315 (9.1%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 82,872 (56.7%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 62,511 (42.8%)
  - matched_generic_only: 23,878 (16.3%)
  - matched_generic_dose: 23,805 (16.3%)
  - matched_atc_dose: 23,227 (15.9%)
  - matched_generic_atc: 11,820 (8.1%)
  - no_generic: 806 (0.6%)
  - matched_drugbank_id: 142 (0.1%)

### Overall
- ESOA ATC coverage: 44,910/146,189 (30.7%)
- ESOA DrugBank coverage: 13,315/146,189 (9.1%)
- ESOA → Drug code coverage: 82,872/146,189 (56.7%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 14:41:33

### Code State
- Branch: master
- Commit: 5c040d0
- Working tree: dirty
  - m dependencies/atcd
  - m dependencies/drugbank_generics
  - m dependencies/fda_ph_scraper
  - M  pipelines/drugs/scripts/lookup.py
  - M  pipelines/drugs/scripts/runners.py

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,035 (83.8%)
- Matched DrugBank ID: 1,804 (74.3%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,035 (83.8%)
  - no_candidates: 222 (9.1%)
  - no_match: 170 (7.0%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 94,997 (65.0%)
- Matched DrugBank ID: 87,782 (60.0%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 78,638 (53.8%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 66,598 (45.6%)
  - matched_atc_dose: 29,922 (20.5%)
  - matched_generic_atc: 18,968 (13.0%)
  - matched_generic_dose: 15,365 (10.5%)
  - matched_generic_only: 8,974 (6.1%)
  - matched_drugbank_id: 5,409 (3.7%)
  - no_generic: 953 (0.7%)

### Overall
- ESOA ATC coverage: 94,997/146,189 (65.0%)
- ESOA DrugBank coverage: 87,782/146,189 (60.0%)
- ESOA → Drug code coverage: 78,638/146,189 (53.8%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 14:59:18

### Code State
- Branch: master
- Commit: 5c040d0
- Working tree: dirty
  - m dependencies/atcd
  - m dependencies/drugbank_generics
  - m dependencies/fda_ph_scraper
  - M  pipelines/drugs/scripts/lookup.py
  - M  pipelines/drugs/scripts/runners.py

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,035 (83.8%)
- Matched DrugBank ID: 1,781 (73.4%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,035 (83.8%)
  - no_candidates: 222 (9.1%)
  - no_match: 170 (7.0%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 96,294 (65.9%)
- Matched DrugBank ID: 88,468 (60.5%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 77,753 (53.2%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 67,478 (46.2%)
  - matched_atc_dose: 28,714 (19.6%)
  - matched_generic_atc: 18,468 (12.6%)
  - matched_generic_dose: 16,281 (11.1%)
  - matched_generic_only: 7,901 (5.4%)
  - matched_drugbank_id: 6,389 (4.4%)
  - no_generic: 958 (0.7%)

### Overall
- ESOA ATC coverage: 96,294/146,189 (65.9%)
- ESOA DrugBank coverage: 88,468/146,189 (60.5%)
- ESOA → Drug code coverage: 77,753/146,189 (53.2%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 15:13:29

### Code State
- Branch: master
- Commit: 5c040d0
- Working tree: dirty
  - m dependencies/atcd
  - m dependencies/drugbank_generics
  - m dependencies/fda_ph_scraper
  - MM pipelines/drugs/scripts/lookup.py
  - M  pipelines/drugs/scripts/runners.py

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,035 (83.8%)
- Matched DrugBank ID: 1,769 (72.9%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,035 (83.8%)
  - no_candidates: 222 (9.1%)
  - no_match: 170 (7.0%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 96,330 (65.9%)
- Matched DrugBank ID: 88,675 (60.7%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 77,679 (53.1%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 67,551 (46.2%)
  - matched_atc_dose: 31,868 (21.8%)
  - matched_generic_atc: 19,604 (13.4%)
  - matched_generic_dose: 13,092 (9.0%)
  - matched_generic_only: 7,834 (5.4%)
  - matched_drugbank_id: 5,281 (3.6%)
  - no_generic: 959 (0.7%)

### Overall
- ESOA ATC coverage: 96,330/146,189 (65.9%)
- ESOA DrugBank coverage: 88,675/146,189 (60.7%)
- ESOA → Drug code coverage: 77,679/146,189 (53.1%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 15:30:57

### Code State
- Branch: master
- Commit: 5c040d0
- Working tree: dirty
  - m dependencies/atcd
  - m dependencies/drugbank_generics
  - m dependencies/fda_ph_scraper
  - MM pipelines/drugs/scripts/lookup.py
  - M  pipelines/drugs/scripts/runners.py

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,070 (85.3%)
- Matched DrugBank ID: 1,822 (75.1%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,070 (85.3%)
  - no_candidates: 195 (8.0%)
  - no_match: 162 (6.7%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 99,641 (68.2%)
- Matched DrugBank ID: 92,095 (63.0%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 77,906 (53.3%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 67,595 (46.2%)
  - matched_atc_dose: 28,214 (19.3%)
  - matched_generic_atc: 18,679 (12.8%)
  - matched_generic_dose: 16,690 (11.4%)
  - matched_drugbank_id: 8,000 (5.5%)
  - matched_generic_only: 6,323 (4.3%)
  - no_generic: 688 (0.5%)

### Overall
- ESOA ATC coverage: 99,641/146,189 (68.2%)
- ESOA DrugBank coverage: 92,095/146,189 (63.0%)
- ESOA → Drug code coverage: 77,906/146,189 (53.3%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 15:47:37

### Code State
- Branch: master
- Commit: 5c040d0
- Working tree: dirty
  - m dependencies/atcd
  - m dependencies/drugbank_generics
  - m dependencies/fda_ph_scraper
  - MM pipelines/drugs/scripts/lookup.py
  - M  pipelines/drugs/scripts/runners.py

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,084 (85.9%)
- Matched DrugBank ID: 1,784 (73.5%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,084 (85.9%)
  - no_candidates: 195 (8.0%)
  - no_match: 148 (6.1%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 99,691 (68.2%)
- Matched DrugBank ID: 88,954 (60.8%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 80,583 (55.1%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 64,916 (44.4%)
  - matched_atc_dose: 32,160 (22.0%)
  - matched_generic_atc: 20,816 (14.2%)
  - matched_generic_dose: 13,992 (9.6%)
  - matched_generic_only: 7,604 (5.2%)
  - matched_drugbank_id: 6,010 (4.1%)
  - no_generic: 690 (0.5%)
  - matched_drugbank_id_dose: 1 (0.0%)

### Overall
- ESOA ATC coverage: 99,691/146,189 (68.2%)
- ESOA DrugBank coverage: 88,954/146,189 (60.8%)
- ESOA → Drug code coverage: 80,583/146,189 (55.1%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

## Run completed 2025-12-03 15:59:24

### Code State
- Branch: master
- Commit: 5c040d0
- Working tree: dirty
  - m dependencies/atcd
  - m dependencies/drugbank_generics
  - m dependencies/fda_ph_scraper
  - M pipelines/drugs/scripts/build_unified_reference.py
  - MM pipelines/drugs/scripts/lookup.py

### Part 1: Prepare Dependencies
- WHO ATC refreshed
- DrugBank lean export refreshed
- FDA brand map rebuilt
- FDA food catalog refreshed
- PNF prepared
- Annex F verified

### Part 2: Match Annex F with ATC/DrugBank IDs
- Total rows: 2,427
- Matched ATC: 2,084 (85.9%)
- Matched DrugBank ID: 1,782 (73.4%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/annex_f_with_atc.csv
- Match reasons:
  - matched: 2,084 (85.9%)
  - no_candidates: 195 (8.0%)
  - no_match: 148 (6.1%)

### Part 3: Match ESOA with ATC/DrugBank IDs
- Total rows: 146,189
- Matched ATC: 99,492 (68.1%)
- Matched DrugBank ID: 89,753 (61.4%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_atc.csv

### Part 4: Bridge ESOA to Annex F Drug Codes
- Total rows: 146,189
- Matched drug codes: 80,329 (54.9%)
- Output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv
- Match reasons:
  - generic_not_in_annex: 65,170 (44.6%)
  - matched_atc_dose: 32,566 (22.3%)
  - matched_generic_atc: 21,632 (14.8%)
  - matched_generic_dose: 13,424 (9.2%)
  - matched_generic_only: 7,433 (5.1%)
  - matched_drugbank_id: 5,273 (3.6%)
  - no_generic: 690 (0.5%)
  - matched_drugbank_id_dose: 1 (0.0%)

### Overall
- ESOA ATC coverage: 99,492/146,189 (68.1%)
- ESOA DrugBank coverage: 89,753/146,189 (61.4%)
- ESOA → Drug code coverage: 80,329/146,189 (54.9%)
- Final output: /Users/carlosresu/github_repos/pids-drg-esoa/outputs/drugs/esoa_with_drug_code.csv

