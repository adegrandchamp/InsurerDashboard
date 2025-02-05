[
    "bene_15",
    "bene_16",
    "bene_17",
    "bene_18",
    "bene_19",
    "bene_20",
    "bene_21",
    "bene_22",
    "bene_23",
    "bene_24",
    "bene_25",
    "clm_carrier",
    "clm_dme",
    "clm_hha",
    "clm_hospice",
    "clm_inpatient",
    "clm_outpatient",
    "clm_snf",
    "pharm_pde"
].forEach((name) =>
    declare({
        databse: "spatial-earth-449020-m3",
        schema: "cms_beneficiary",
        name,
    })
);