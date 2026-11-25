# Data

The three data sets are taken from the R package **CDM** (Robitzsch, Kiefer, George and Ünlü), version 8.3-14 (2025), distributed under the GNU General Public License (version 2 or later): <https://CRAN.R-project.org/package=CDM>. They were converted from the package's `.rda` files to CSV without any other change. They are redistributed here under the same licence.

| File | Content | CDM object | Original sources |
|---|---|---|---|
| `fraction_data.csv` | 536 students × 20 fraction-subtraction items (1 = correct) | `fraction.subtraction.data` | Tatsuoka (1984); Tatsuoka (2002) |
| `fraction_qmatrix.csv` | Q-matrix, 20 items × 8 attributes | `fraction.subtraction.qmatrix` | de la Torre & Douglas (2004) |
| `ecpe_data.csv` | 2922 examinees × 28 ECPE grammar items | `data.ecpe$data` (column `id` removed) | Templin & Hoffman (2013) |
| `ecpe_qmatrix.csv` | Q-matrix, 28 items × 3 attributes (morphosyntactic, cohesive, lexical) | `data.ecpe$q.matrix` | Templin & Hoffman (2013) |
| `timss_data.csv` | 757 US eighth graders × 23 TIMSS 2003 mathematics items (booklet 1) | `data.timss03.G8.su$data` (columns `idstud`, `idbook` removed) | Su et al. (2013) |
| `timss_qmatrix.csv` | Q-matrix, 23 items × 13 attributes S1–S13 | `data.timss03.G8.su$q.matrix` | Su et al. (2013) |
| `timss_iteminfo.csv` | TIMSS item descriptions | `data.timss03.G8.su$iteminfo` | Su et al. (2013) |

In the paper, items are labelled I1–I20, E1–E28 and T1–T23, in the column order of these files.

## References

- de la Torre, J., & Douglas, J. A. (2004). Higher-order latent trait models for cognitive diagnosis. *Psychometrika*, 69(3), 333–353.
- Su, Y.-L., Choi, K. M., Lee, W.-C., Choi, T., & McAninch, M. (2013). *Hierarchical cognitive diagnostic analysis for TIMSS 2003 mathematics* (CASMA Research Report 35). University of Iowa.
- Tatsuoka, C. (2002). Data analytic methods for latent partially ordered classification models. *Journal of the Royal Statistical Society: Series C*, 51(3), 337–350.
- Tatsuoka, K. K. (1984). *Analysis of errors in fraction addition and subtraction problems* (Final Report for Grant No. NIE-G-81-0002). University of Illinois.
- Templin, J., & Hoffman, L. (2013). Obtaining diagnostic classification model estimates using Mplus. *Educational Measurement: Issues and Practice*, 32(2), 37–50.
