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

# Source
dataset <- drugbank

# Canonical generic names
generics <- dataset$drugs$general_information %>%
  select(drugbank_id, generic = name)

# Synonyms
synonyms <- dataset$drugs$synonyms %>%
  select(drugbank_id, synonym)

# Brand blacklist from DrugBank brands and non-generic products
brand_names <- unique(c(
  dataset$drugs$international_brands$brand,
  dataset$products %>% filter(tolower(generic) == "false") %>% pull(name)
))

# Build one-column list of generic names & synonyms, de-branded, de-chemicalized, no parentheses
generic_synonyms <- generics %>%
  left_join(synonyms, by = "drugbank_id") %>%
  rowwise() %>%
  mutate(all_names = list(unique(na.omit(c(generic, synonym))))) %>%
  ungroup() %>%
  select(all_names) %>%
  unnest_longer(all_names, values_to = "all_names") %>%
  mutate(all_names = trimws(all_names)) %>%
  filter(!is.na(all_names), all_names != "") %>%
  # drop brand names
  filter(!(all_names %in% brand_names)) %>%
  # remove chemical-style names and parentheses
  filter(
    !grepl("[-+:,()/]", all_names), # excludes parentheses and symbols
    !grepl("[Nn][[:digit:]]", all_names), # removes N2, n3, etc.
    !grepl("[[:digit:]]", all_names) # removes numeric chemical-like names
  ) %>%
  distinct(all_names) %>%
  arrange(all_names) %>%
  transmute(name = all_names)

# Write CSV (single column)
write.csv(generic_synonyms, output_path, row.names = FALSE, quote = TRUE)
