# EDA summary

## Dataset
- Rows: 4177
- Features: 8 (+ target `Age`)

## Data quality
- Missing values in raw data: rare (~1–2% in `Diameter`, `Whole_weight`, `Shell_weight`); imputed with median
- Zero `Height` values: replaced with median height
- `Sex` values normalized (`f` → `F`)

## Target
- Age mean: 11.43 years
- Age std: 3.22 years
- Age range: [2.5, 30.5]

## Relationships with age (Pearson correlation)
- Strongest: `Shell_weight` (0.62), `Diameter` (0.56), `Height` / `Length` (0.56)
- Moderate: `Whole_weight` (0.53), `Viscera_weight` (0.50), `Shucked_weight` (0.42)
- Median age by sex: M 11.5, F 11.5, I (infant) 9.5

## Notes
- Age is derived as `Rings + 1.5`.
- Morphology and weight features are positively associated with age; infants are systematically younger.
- Several numeric features are skewed; tree-based models handle this reasonably without heavy outlier removal.
