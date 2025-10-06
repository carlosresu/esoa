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

# Canonical generic names only, normalized to lowercase
generic_names <- dataset$drugs$general_information %>%
  transmute(name = tolower(trimws(name))) %>%
  filter(!is.na(name), name != "") %>%
  distinct(name) %>%
  arrange(name)

# Write CSV (single column)
write.csv(generic_names, output_path, row.names = FALSE, quote = TRUE)
