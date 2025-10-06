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

# Source
dataset <- drugbank

# Synonyms column filtered to English (case-insensitive, allows combined strings)
english_synonyms <- dataset$drugs$synonyms %>%
  mutate(language_lower = tolower(language)) %>%
  filter(!is.na(language_lower), str_detect(language_lower, "english")) %>%
  transmute(name = normalize_name(synonym)) %>%
  filter(!is.na(name), name != "") %>%
  filter(!str_detect(name, "[-+(),'/]")) %>%
  distinct(name) %>%
  arrange(name)

# Write CSV (single column)
write.csv(english_synonyms, output_path, row.names = FALSE, quote = TRUE)
