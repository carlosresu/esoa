ensure_installed <- function(packages, repos = "https://cloud.r-project.org") {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg, repos = repos)
    }
  }
}

get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  }
  getwd()
}

ensure_installed(c("tidyverse", "devtools"))
if (!requireNamespace("dbdataset", quietly = TRUE)) {
  devtools::install_github("interstellar-Consultation-Services/dbdataset", quiet = TRUE)
}

library(tidyverse)
library(dbdataset)

script_dir <- get_script_dir()
output_dir <- file.path(script_dir, "output")
output_path <- file.path(output_dir, "generics.csv")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

normalize_name <- function(values) {
  trimmed <- trimws(values)
  transliterated <- iconv(trimmed, from = "", to = "ASCII//TRANSLIT")
  fallback_needed <- is.na(transliterated)
  if (any(fallback_needed)) {
    transliterated[fallback_needed] <- trimmed[fallback_needed]
  }
  tolower(transliterated)
}

# --- smarter combo detection & formatting to "; " ---
standardize_combo_generic <- function(x) {
  # 1) keep original for tokenization (we want delimiters), then normalize tokens
  raw <- x %||% ""
  # unify common separators to a single token "|"
  unified <- raw %>%
    # protect hyphenated words (don’t split on hyphen)
    gsub("\\s+(and|with|plus)\\s+", "|", ., ignore.case = TRUE) %>%
    gsub("\\s*/\\s*", "|", ., perl = TRUE) %>%
    gsub("\\s*,\\s*", "|", ., perl = TRUE) %>%
    gsub("\\s*\\+\\s*", "|", ., perl = TRUE) %>%
    gsub("\\s*;\\s*", "|", ., perl = TRUE)
  
  parts <- strsplit(unified, "\\|", fixed = FALSE)[[1]]
  if (length(parts) == 0) parts <- raw
  
  # normalize each token to only a-z and spaces, drop empties, de-duplicate
  norm_tokens <- parts %>%
    normalize_name() %>%
    str_replace_all("[^a-z ]", " ") %>%
    str_squish() %>%
    discard(~ .x == "")
  
  norm_tokens <- unique(norm_tokens)
  
  if (length(norm_tokens) == 0) return(NA_character_)
  if (length(norm_tokens) == 1) return(norm_tokens[[1]])
  
  # sort for deterministic output, then join with "; "
  paste(sort(norm_tokens), collapse = "; ")
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# ================================
# Source
# ================================
dataset <- drugbank

# ================================
# Build a generic-name map with SMART combo detection.
# We purposefully collect "raw" names (with delimiters) first, then
# standardize to a-z + spaces and join combos with "; ".
# ================================

# 1) Primary names
gi_raw <- dataset$drugs$general_information %>%
  transmute(drugbank_id, raw_name = name)

# 2) English synonyms (raw)
syn_raw <- dataset$drugs$synonyms %>%
  mutate(language_lower = tolower(language)) %>%
  filter(!is.na(language_lower), str_detect(language_lower, "english")) %>%
  transmute(drugbank_id, raw_name = synonym)

# 3) Mixtures (explicit ingredient lists like "X + Y")
mix_raw <- dataset$drugs$mixtures %>%
  transmute(drugbank_id, raw_name = ingredients)

# Optionally: products names can be very branded; skip to avoid noise.

# Combine all raw sources
all_names_raw <- bind_rows(gi_raw, syn_raw, mix_raw) %>%
  filter(!is.na(raw_name), raw_name != "")

# Standardize generics with combo detection → "; "-delimited when multiple
generic_name_map <- all_names_raw %>%
  mutate(generic = standardize_combo_generic(raw_name)) %>%
  filter(!is.na(generic), generic != "") %>%
  # final guard: only allow a-z, semicolon, spaces
  filter(str_detect(generic, "^[a-z; ]+$")) %>%
  distinct(drugbank_id, generic)

# ================================
# ATC codes (per DrugBank ID)
# ================================
atc_tbl <- dataset$drugs$atc_codes %>%
  transmute(drugbank_id, atc_code = atc_code) %>%
  distinct()

# ================================
# Route/Form/Dose candidates
# Prefer dosages (structured). Also harvest from products. KEEP provenance to score support.
# ================================
dosages_tbl <- dataset$drugs$dosages %>%
  transmute(
    drugbank_id,
    route   = na_if(str_squish(route), ""),
    form    = na_if(str_squish(form), ""),
    dosage  = na_if(str_squish(str_to_lower(strength)), ""),
    source  = "DOSAGES",
    country = NA_character_,
    fda_application_number = NA_character_,
    ndc_product_code       = NA_character_,
    dpd_id                 = NA_character_,
    ema_product_code       = NA_character_,
    ema_ma_number          = NA_character_
  ) %>%
  distinct()

products_tbl <- dataset$products %>%
  transmute(
    drugbank_id,
    route   = na_if(str_squish(route), ""),
    form    = na_if(str_squish(dosage_form), ""),
    dosage  = na_if(str_squish(str_to_lower(strength)), ""),
    source,
    country,
    fda_application_number = na_if(str_squish(fda_application_number), ""),
    ndc_product_code       = na_if(str_squish(ndc_product_code), ""),
    dpd_id                 = na_if(str_squish(dpd_id), ""),
    ema_product_code       = na_if(str_squish(ema_product_code), ""),
    ema_ma_number          = na_if(str_squish(ema_ma_number), "")
  ) %>%
  filter(!is.na(route) | !is.na(form)) %>%
  distinct()

route_form <- bind_rows(dosages_tbl, products_tbl) %>%
  distinct(
    drugbank_id, route, form, dosage,
    source, country, fda_application_number, ndc_product_code,
    dpd_id, ema_product_code, ema_ma_number
  )

# ================================
# Build per-id expansions (chunked) to keep memory bounded.
# Cross generics × atcs × (route, form, dosage, provenance) for each id.
# ================================
ids <- atc_tbl %>%
  distinct(drugbank_id) %>%
  pull(drugbank_id)

expand_one_id <- function(id_chr) {
  gens <- generic_name_map %>%
    filter(drugbank_id == id_chr) %>%
    distinct(generic) %>%
    mutate(dummy = 1L)
  if (nrow(gens) == 0) return(NULL)
  
  atcs <- atc_tbl %>%
    filter(drugbank_id == id_chr) %>%
    distinct(atc_code) %>%
    mutate(dummy = 1L)
  if (nrow(atcs) == 0) return(NULL)
  
  rf <- route_form %>%
    filter(drugbank_id == id_chr) %>%
    distinct(
      route, form, dosage,
      source, country, fda_application_number, ndc_product_code,
      dpd_id, ema_product_code, ema_ma_number
    )
  if (nrow(rf) == 0) {
    rf <- tibble(
      route = NA_character_,
      form  = NA_character_,
      dosage = NA_character_,
      source = NA_character_,
      country = NA_character_,
      fda_application_number = NA_character_,
      ndc_product_code = NA_character_,
      dpd_id = NA_character_,
      ema_product_code = NA_character_,
      ema_ma_number = NA_character_
    )
  }
  
  ga <- merge(gens, atcs, by = "dummy", all = TRUE, allow.cartesian = TRUE) %>%
    select(-dummy)
  out <- merge(ga, rf, all = TRUE, allow.cartesian = TRUE)
  
  out$drugbank_id <- id_chr
  out %>%
    select(
      generic, route, form, atc_code, dosage, drugbank_id,
      source, country, fda_application_number, ndc_product_code,
      dpd_id, ema_product_code, ema_ma_number
    ) %>%
    distinct()
}

final_chunks <- purrr::map(ids, expand_one_id)
final_long <- bind_rows(final_chunks) %>%
  mutate(
    route  = na_if(str_squish(route), ""),
    form   = na_if(str_squish(form), ""),
    dosage = na_if(str_squish(dosage), "")
  )

# ================================
# Clinically meaningful ATC selection per (generic[s], route, form, dosage)
# Rules:
#  - If generic represents a COMBINATION (detected via tokens), prefer COMBO ATCs (suffix >= 50 or J05AR*).
#  - If generic is SINGLE-INGREDIENT, prefer MONO ATCs; within mono prefer base “…01” if present.
#  - Among remaining ties, choose the ATC with the strongest product/regulatory SUPPORT for that tuple.
#  - Final tie-breakers: smaller numeric suffix, then lexicographic (deterministic).
# ================================

# Flag ATC "combination" by code (WHO convention and J05AR family)
final_long <- final_long %>%
  mutate(
    last2 = suppressWarnings(as.integer(stringr::str_sub(atc_code, -2))),
    is_atc_combo = stringr::str_detect(coalesce(atc_code, ""), "^J05AR") |
      (!is.na(last2) & last2 >= 50),
    # detect combinations from standardized generic (now "; " delimited if combo)
    is_combo_generic = stringr::str_detect(coalesce(generic, ""), ";")
  )

# Build a support key from provenance fields (what backs this tuple+ATC in the underlying product/dosage data)
support_tbl <- final_long %>%
  mutate(
    support_key = paste(
      coalesce(source, ""),
      coalesce(country, ""),
      coalesce(fda_application_number, ""),
      coalesce(ndc_product_code, ""),
      coalesce(dpd_id, ""),
      coalesce(ema_product_code, ""),
      coalesce(ema_ma_number, ""),
      sep = "|"
    ),
    support_key = dplyr::na_if(support_key, "||||||")
  ) %>%
  group_by(generic, route, form, dosage, atc_code, is_atc_combo, is_combo_generic) %>%
  summarize(
    support_count = n_distinct(support_key, na.rm = TRUE),
    .groups = "drop"
  )

# Rank ATCs within each clinical tuple using the rules above
best_atc <- support_tbl %>%
  mutate(
    # primary preference: combo fit
    pref_combo = dplyr::case_when(
      is_combo_generic &  is_atc_combo ~ 1L,  # combo generic → combo ATC
      !is_combo_generic & !is_atc_combo ~ 1L, # single generic → mono ATC
      TRUE ~ 2L
    ),
    # secondary: for single-ingredient tuples, prefer “…01” as base mono code
    suffix = suppressWarnings(as.integer(stringr::str_sub(atc_code, -2))),
    pref_base_mono = dplyr::case_when(
      !is_combo_generic & !is_atc_combo & !is.na(suffix) & suffix == 1 ~ 1L,
      !is_combo_generic & !is_atc_combo ~ 2L,
      TRUE ~ 3L
    )
  ) %>%
  group_by(generic, route, form, dosage) %>%
  arrange(
    pref_combo,
    pref_base_mono,
    dplyr::desc(support_count), # stronger backing first
    suffix,                     # smaller suffix (e.g., 01 before 09)
    atc_code                    # deterministic final tie-break
  ) %>%
  slice_head(n = 1) %>%
  ungroup()

# Optional: gather all DrugBank IDs that contributed evidence for the chosen ATC per tuple
ids_for_best <- final_long %>%
  semi_join(best_atc, by = c("generic","route","form","dosage","atc_code")) %>%
  group_by(generic, route, form, dosage, atc_code) %>%
  summarize(
    drugbank_id = paste(sort(unique(na.omit(drugbank_id))), collapse = ";"),
    .groups = "drop"
  )

# Final table: one clinically-preferred ATC per (generic[s], route, form, dosage)
final_best <- best_atc %>%
  select(generic, route, form, atc_code, dosage) %>%
  left_join(ids_for_best, by = c("generic","route","form","dosage","atc_code")) %>%
  arrange(generic, route, form, atc_code, dosage)

# Write out
write.csv(final_best, output_path, row.names = FALSE, quote = TRUE)