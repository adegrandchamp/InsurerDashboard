[
    "icd_diagnoses",
    "seed_months_of_year",
    "seed_months_numeric",
    "seed_years",
    "ssa_to_fips_crosswalk",
    "state_code_lookup",
    "zip_to_fips", 
    "bic_lookup",
    "npi_taxonomy",
    "npi_full_list",
    "clm_carr_names"
].forEach((name) =>
    declare({
        databse: "spatial-earth-449020-m3",
        schema: "cms_supplemental",
        name,
    })
);