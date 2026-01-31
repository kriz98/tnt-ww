# preprocess.R
# Minimal preprocessing script
# - Assumes you provide de-identified CSVs locally under ./data
# - Produces harmonized one-hot encoded datasets for TabPFN/AutoTabPFN

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(naniar)
  library(stringr)
  library(fastDummies)
})

# -------------------------
# User-configurable settings
# -------------------------
DEV_IN  <- "data/dev_raw.csv"   # development cohort (local)
EXT_IN  <- "data/ext_raw.csv"   # external cohort (local)

DEV_OUT <- "data/dev_encoded.csv"
EXT_OUT <- "data/ext_encoded.csv"

# Outcome column names (edit to match your files)
OUTCOME_DEV <- "outcome_dev"    # e.g., p_complete_response
OUTCOME_EXT <- "outcome_ext"    # e.g., persistent_cCR

# Sentinel missing values to convert to NA (edit as needed)
SENTINELS <- c(9999, "9999", 999999, "999999", "999999.0", "Unknown")

# -------------------------
# 1) Read + standard cleanup
# -------------------------
read_clean <- function(path) {
  readr::read_csv(path, show_col_types = FALSE) %>%
    janitor::clean_names() %>%
    mutate(across(where(~ is.character(.x) || is.factor(.x)),
                  ~ str_trim(as.character(.x)))) %>%
    naniar::replace_with_na_all(condition = ~ .x %in% SENTINELS)
}

dev <- read_clean(DEV_IN)
ext <- read_clean(EXT_IN)

# -------------------------------------------------------
# 2) Harmonize variable coding (edit this to your schema)
# -------------------------------------------------------
# This section is intentionally minimal and **template-like**.
# Add/modify recodes so dev + ext use the same levels/labels.

harmonize_common <- function(df) {
  df %>%
    mutate(
      # Example standardizations (edit/remove as needed)
      gender = case_when(
        gender %in% c("M", "Male", "male") ~ "Male",
        gender %in% c("F", "Female", "female") ~ "Female",
        TRUE ~ gender
      ),
      emvi = case_when(
        emvi %in% c("Yes", "1", 1) ~ "Yes",
        emvi %in% c("No", "0", 0) ~ "No",
        TRUE ~ emvi
      ),
      clinical_t_stage = case_when(
        clinical_t_stage %in% c("T4A", "T4B", "4a", "4b", "4A", "4B") ~ "4",
        clinical_t_stage %in% c("T3", "3a", "3b", "3c", "3d", "3A", "3B", "3C", "3D") ~ "3",
        clinical_t_stage %in% c("T2", "2") ~ "2",
        clinical_t_stage %in% c("T1", "1") ~ "1",
        TRUE ~ as.character(clinical_t_stage)
      ),
      clinical_n_stage = case_when(
        clinical_n_stage %in% c("1", "1a", "1b", "1c") ~ "1",
        clinical_n_stage %in% c("2", "2a", "2b") ~ "2",
        clinical_n_stage %in% c("0") ~ "0",
        TRUE ~ as.character(clinical_n_stage)
      )
    )
}

dev <- dev %>% harmonize_common()
ext <- ext %>% harmonize_common()

# -------------------------------------------------------
# 3) Restrict to predictors + outcomes (edit as needed)
# -------------------------------------------------------
# Keep only columns you intend to model with, plus the outcome.
# If you already curated columns upstream, you can leave this as-is.
keep_dev <- intersect(names(dev), c(names(dev))) %>% unique()
keep_ext <- intersect(names(ext), c(names(ext))) %>% unique()

dev <- dev %>% select(all_of(keep_dev))
ext <- ext %>% select(all_of(keep_ext))

# Ensure outcomes exist
stopifnot(OUTCOME_DEV %in% names(dev))
stopifnot(OUTCOME_EXT %in% names(ext))

# -------------------------
# 4) One-hot encode predictors
# -------------------------
# Identify categorical predictors (exclude outcomes)
cat_dev <- dev %>%
  select(where(~ is.character(.x) || is.factor(.x))) %>%
  names() %>%
  setdiff(OUTCOME_DEV)

cat_ext <- ext %>%
  select(where(~ is.character(.x) || is.factor(.x))) %>%
  names() %>%
  setdiff(OUTCOME_EXT)

dev_enc <- dev %>%
  fastDummies::dummy_cols(
    select_columns = cat_dev,
    remove_selected_columns = TRUE,
    remove_first_dummy = FALSE
  )

ext_enc <- ext %>%
  fastDummies::dummy_cols(
    select_columns = cat_ext,
    remove_selected_columns = TRUE,
    remove_first_dummy = FALSE
  )

# ---------------------------------------------
# 5) Harmonize feature columns across both sets
# ---------------------------------------------
features <- union(names(dev_enc), names(ext_enc)) %>%
  setdiff(c(OUTCOME_DEV, OUTCOME_EXT))

# Add any missing columns (filled with 0)
for (col in features) {
  if (!col %in% names(dev_enc)) dev_enc[[col]] <- 0
  if (!col %in% names(ext_enc)) ext_enc[[col]] <- 0
}

# Reorder: features first, then outcomes
dev_enc <- dev_enc %>% select(all_of(features), all_of(OUTCOME_DEV))
ext_enc <- ext_enc %>% select(all_of(features), all_of(OUTCOME_EXT))

# Safety checks
stopifnot(setequal(features, setdiff(names(dev_enc), OUTCOME_DEV)))
stopifnot(setequal(features, setdiff(names(ext_enc), OUTCOME_EXT)))

# -------------------------
# 6) Write encoded outputs
# -------------------------
dir.create("data", showWarnings = FALSE)
readr::write_csv(dev_enc, DEV_OUT)
readr::write_csv(ext_enc, EXT_OUT)

message("Wrote:\n - ", DEV_OUT, "\n - ", EXT_OUT)
