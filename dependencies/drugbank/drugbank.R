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

# ================================
# Source
# ================================
dataset <- drugbank

# ================================
# Canonical generic names + English synonyms (normalized).
# Filter generics to ONLY a-z and spaces (no digits/punct).
# ================================
drugs_gi <- dataset$drugs$general_information %>%
  transmute(
    drugbank_id,
    generic = normalize_name(name)
  ) %>%
  filter(!is.na(generic), generic != "", str_detect(generic, "^[a-z ]+$"))

synonyms_en <- dataset$drugs$synonyms %>%
  mutate(language_lower = tolower(language)) %>%
  filter(!is.na(language_lower), str_detect(language_lower, "english")) %>%
  transmute(
    drugbank_id,
    generic = normalize_name(synonym)
  ) %>%
  filter(!is.na(generic), generic != "", str_detect(generic, "^[a-z ]+$"))

generic_name_map <- bind_rows(drugs_gi, synonyms_en) %>%
  distinct(drugbank_id, generic)

# ================================
# ATC codes (per DrugBank ID)
# ================================
atc_tbl <- dataset$drugs$atc_codes %>%
  transmute(drugbank_id, atc_code = atc_code) %>%
  distinct()

# ================================
# Route/Form/Dose candidates
# Prefer dosages (structured). Also harvest from products, but SKIP brands entirely.
# ================================
dosages_tbl <- dataset$drugs$dosages %>%
  transmute(
    drugbank_id,
    route  = na_if(str_squish(route), ""),
    form   = na_if(str_squish(form), ""),
    dosage = na_if(str_squish(str_to_lower(str_squish(strength))), "")
  ) %>%
  distinct()

products_tbl <- dataset$products %>%
  transmute(
    drugbank_id,
    route  = na_if(str_squish(route), ""),
    form   = na_if(str_squish(dosage_form), ""),
    dosage = na_if(str_squish(str_to_lower(strength)), "")
  ) %>%
  filter(!is.na(route) | !is.na(form)) %>%
  distinct()

route_form <- bind_rows(dosages_tbl, products_tbl) %>%
  distinct(drugbank_id, route, form, dosage)

# ================================
# Build per-id expansions in chunks to avoid memory blow-ups.
# Cross generics × atcs × (route, form, dosage) for each id.
# ================================
ids <- atc_tbl %>%
  distinct(drugbank_id) %>%
  pull(drugbank_id)

expand_one_id <- function(id_chr) {
  gens <- generic_name_map %>%
    filter(drugbank_id == id_chr) %>%
    distinct(generic) %>%
    mutate(dummy = 1L)
  if (nrow(gens) == 0) {
    return(NULL)
  }

  atcs <- atc_tbl %>%
    filter(drugbank_id == id_chr) %>%
    distinct(atc_code) %>%
    mutate(dummy = 1L)
  if (nrow(atcs) == 0) {
    return(NULL)
  }

  rf <- route_form %>%
    filter(drugbank_id == id_chr) %>%
    distinct(route, form, dosage)
  if (nrow(rf) == 0) {
    rf <- tibble(route = NA_character_, form = NA_character_, dosage = NA_character_)
  }

  ga <- merge(gens, atcs, by = "dummy", all = TRUE, allow.cartesian = TRUE) %>%
    select(-dummy)
  out <- merge(ga, rf, all = TRUE, allow.cartesian = TRUE)

  out$drugbank_id <- id_chr
  out %>%
    select(generic, route, form, atc_code, dosage, drugbank_id) %>%
    distinct()
}

final_chunks <- purrr::map(ids, expand_one_id)
final_long <- bind_rows(final_chunks)

# ================================
# Dose column rule:
# Keep dosage only if within SAME (drugbank_id, route, form) there are multiple ATC codes
# and dose helps disambiguate; otherwise blank it.
# ================================
final_long <- final_long %>%
  mutate(
    route  = na_if(str_squish(route), ""),
    form   = na_if(str_squish(form), ""),
    dosage = na_if(str_squish(dosage), "")
  )

dose_mask <- final_long %>%
  group_by(drugbank_id, route, form) %>%
  summarize(
    atc_per_group = n_distinct(atc_code),
    any_dose = any(!is.na(dosage)),
    .groups = "drop"
  )

final_long <- final_long %>%
  left_join(dose_mask, by = c("drugbank_id", "route", "form")) %>%
  mutate(
    dosage = if_else(!is.na(atc_per_group) & atc_per_group > 1 & any_dose, dosage, NA_character_)
  ) %>%
  select(-atc_per_group, -any_dose) %>%
  arrange(generic, route, form, atc_code, dosage, drugbank_id) %>%
  distinct()

# ================================
# Write CSV (single long form): generic, route, form, atc_code, dosage, drugbank_id
# ================================
write.csv(final_long, output_path, row.names = FALSE, quote = TRUE)
